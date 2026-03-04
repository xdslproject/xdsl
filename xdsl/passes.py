from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    NamedTuple,
)

from typing_extensions import Self, TypeVar

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.utils.arg_spec import (
    ArgSpec,
    ArgSpecConvertible,
    parse_pipeline,
)
from xdsl.utils.hints import type_repr

ModulePassT = TypeVar("ModulePassT", bound="ModulePass")


@dataclass(frozen=True)
class ModulePass(ArgSpecConvertible):
    """
    A Pass is a named rewrite pass over an IR module that can accept arguments.

    All passes are expected to leave the IR in a valid state *after* application,
    meaning that a call to .verify() succeeds on the whole module. In turn, all
    passes can expect that the IR they are applied to is in a valid state. It
    is not required that the IR verifies at any point while the pass is being
    applied.

    In order to make a pass accept arguments, it must be a dataclass. Furthermore,
    only the following types are supported as argument types:

    Base types:                int | float | bool | string
    N-tuples of base types:
        tuple[int, ...], tuple[int|float, ...], tuple[int, ...] | tuple[float, ...]
    Top-level optional:        ... | None

    Pass arguments on the CLI are formatted as follows:

    CLI arg                             Mapped to class field
    -------------------------           ------------------------------
    my-pass{arg-1=1}                    arg_1: int             = 1
    my-pass{arg-1}                      arg_1: int | None      = None
    my-pass{arg-1=1,2,3}                arg_1: tuple[int, ...] = (1, 2, 3)
    my-pass{arg-1=true}                 arg_1: bool | None     = True
    """

    name: ClassVar[str]

    @abstractmethod
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None: ...

    def apply_to_clone(
        self, ctx: Context, op: builtin.ModuleOp
    ) -> tuple[Context, builtin.ModuleOp]:
        """
        Creates deep copies of the module and the context, and returns the result of
        calling `apply` on them.
        """
        ctx = ctx.clone()
        op = op.clone()
        self.apply(ctx, op)
        return ctx, op

    @classmethod
    def from_pass_spec(cls, spec: ArgSpec) -> Self:
        """Alias for ``from_spec`` for backward compatibility."""
        return cls.from_spec(spec)

    def pipeline_pass_spec(self, *, include_default: bool = False) -> ArgSpec:
        """Alias for ``spec`` for backward compatibility."""
        return self.spec(include_default=include_default)

    @classmethod
    def schedule_space(
        cls, ctx: Context, module_op: builtin.ModuleOp
    ) -> tuple[Self, ...]:
        """
        Returns a tuple of `Self` that can be applied to rewrite the given module with
        the given context without error.
        The default implementation attempts to construct an instance with no parameters,
        and run it on the module_op; if the module_op is mutated then the pass instance
        is returned.
        Parametrizable passes should override this implementation to provide a full
        schedule space of transformations.
        """
        try:
            pass_instance = cls()
            _, cloned_module = pass_instance.apply_to_clone(ctx, module_op)
            if module_op.is_structurally_equivalent(cloned_module):
                return ()
        except Exception:
            return ()
        return (pass_instance,)


class PassOptionInfo(NamedTuple):
    """The name, expected type, and default value for one option of a module pass."""

    name: str
    expected_type: str
    default_value: str | None = None


def get_pass_option_infos(
    arg: type[ModulePassT],
) -> tuple[PassOptionInfo, ...]:
    """
    Returns the expected argument names, types, and optional expected values for options
    for the given pass.
    """

    return tuple(
        PassOptionInfo(
            field.name,
            type_repr(field.type),
            str(getattr(arg, field.name)).lower() if hasattr(arg, field.name) else None,
        )
        for field in dataclasses.fields(arg)
    )


@dataclass(frozen=True)
class PassPipeline:
    """
    A representation of a pass pipeline, with an optional callback to be executed
    between each of the passes.
    """

    passes: tuple[ModulePass, ...]
    """
    These will be executed sequentially during the execution of the pipeline.
    """
    callback: Callable[[ModulePass, builtin.ModuleOp, ModulePass], None] | None = field(
        default=None
    )
    """
    Function called in between every pass, taking the pass that just ran, the module,
    and the next pass.
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if not self.passes:
            # Early exit to avoid fetching a non-existing last pass.
            return
        callback = self.callback

        for prev, next in zip(self.passes[:-1], self.passes[1:]):
            prev.apply(ctx, op)
            if callback is not None:
                callback(prev, op, next)

        self.passes[-1].apply(ctx, op)

    @staticmethod
    def parse_spec(
        available_passes: dict[str, Callable[[], type[ModulePass]]],
        spec: str,
        callback: Callable[[ModulePass, builtin.ModuleOp, ModulePass], None]
        | None = None,
    ) -> PassPipeline:
        specs = tuple(parse_pipeline(spec))
        unrecognised_passes = tuple(
            p.name for p in specs if p.name not in available_passes
        )
        if unrecognised_passes:
            raise ValueError(f"Unrecognized passes: {list(unrecognised_passes)}")

        passes = tuple(available_passes[p.name]().from_pass_spec(p) for p in specs)

        return PassPipeline(passes, callback)
