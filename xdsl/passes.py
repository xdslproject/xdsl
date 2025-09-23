from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import Field, dataclass, field
from types import NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    NamedTuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import Self, TypeVar

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.utils.hints import isa, type_repr
from xdsl.utils.parse_pipeline import (
    PassArgElementType,
    PassArgListType,
    PipelinePassSpec,
    parse_pipeline,
)

ModulePassT = TypeVar("ModulePassT", bound="ModulePass")


@dataclass(frozen=True)
class ModulePass(ABC):
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
    def from_pass_spec(cls, spec: PipelinePassSpec) -> Self:
        """
        This method takes a PipelinePassSpec, does type checking on the
        arguments, and then instantiates an instance of the ModulePass
        from the spec.
        """
        if spec.name != cls.name:
            raise ValueError(
                f"Cannot create Pass {cls.name} from pass arguments for pass {spec.name}"
            )

        # normalize spec arg names:
        spec_arguments_dict: dict[str, PassArgListType] = (
            spec.normalize_arg_names().args
        )

        # get all dataclass fields
        fields: tuple[Field[Any], ...] = dataclasses.fields(cls)

        # start constructing the argument dict for the dataclass
        arg_dict = dict[str, PassArgListType | PassArgElementType | None]()

        required_fields = cls.required_fields()

        field_types = get_type_hints(cls)

        # iterate over all fields of the dataclass
        for op_field in fields:
            # ignore the name field and everything that's not used by __init__
            if op_field.name == "name" or not op_field.init:
                continue
            # check that non-optional fields are present
            if op_field.name not in spec_arguments_dict:
                if op_field.name not in required_fields:
                    arg_dict[op_field.name] = _get_default(op_field)
                    continue
                raise ValueError(f'Pass {cls.name} requires argument "{op_field.name}"')

            # convert pass arg to the correct type:
            field_type = field_types[op_field.name]
            arg_dict[op_field.name] = _convert_pass_arg_to_type(
                spec_arguments_dict.pop(op_field.name),
                field_type,
            )
            # we use .pop here to also remove the arg from the dict

        # if not all args were removed we raise an error
        if len(spec_arguments_dict) != 0:
            arguments_str = ", ".join(f'"{arg}"' for arg in spec_arguments_dict)
            fields_str = ", ".join(f'"{field.name}"' for field in fields)
            raise ValueError(
                f"Provided arguments [{arguments_str}] not found in expected pass "
                f"arguments [{fields_str}]"
            )

        # instantiate the dataclass using kwargs
        return cls(**arg_dict)

    @classmethod
    def required_fields(cls) -> set[str]:
        """
        Inspects the definition of the pass for fields that do not have default values.
        """
        return {
            field.name for field in dataclasses.fields(cls) if not _is_optional(field)
        }

    def pipeline_pass_spec(self, *, include_default: bool = False) -> PipelinePassSpec:
        """
        This function takes a ModulePass and returns a PipelinePassSpec.

        If `include_default` is `True`, then optional arguments are not included in the
        spec.
        """
        # get all dataclass fields
        fields = dataclasses.fields(self)
        args: dict[str, PassArgListType] = {}

        # iterate over all fields of the dataclass
        for op_field in fields:
            name = op_field.name
            # ignore the name field and everything that's not used by __init__
            if name == "name" or not op_field.init:
                continue

            val = getattr(self, name)

            if _is_optional(op_field):
                if val == _get_default(op_field) and not include_default:
                    continue

            if val is None:
                arg_list = ()
            elif isinstance(val, PassArgElementType):
                arg_list = (val,)
            else:
                arg_list = val

            args[name] = arg_list
        return PipelinePassSpec(self.name, args)

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

    def __str__(self) -> str:
        return str(self.pipeline_pass_spec())


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


def _convert_pass_arg_to_type(
    value: PassArgListType, dest_type: Any
) -> PassArgListType | PassArgElementType | None:
    """
    Takes in a list of pass args, and converts them to the required type.

    value,      dest_type,      result
    []          int | None      None
    [1]         int | None      1
    [1]         tuple[int, ...] (1,)
    [1,2]       tuple[int, ...] (1,2)
    [1,2]       int | None      Error
    []          int             Error

    And so on
    """
    origin = get_origin(dest_type)

    # we need to special case optionals as [] means no option given
    if origin in [Union, UnionType]:
        if len(value) == 0:
            if NoneType in get_args(dest_type):
                return None
            else:
                raise ValueError("Argument must contain a value")

    # first check if an individual value passes the type check
    if len(value) == 1 and isa(value[0], dest_type):
        return value[0]

    # then check if n-tuple value is okay
    if isa(value, dest_type):
        return value

    # at this point we exhausted all possibilities
    raise ValueError(f"Incompatible types: given {value}, expected {dest_type}")


def _is_optional(field: Field[Any]):
    """
    Shorthand to check if the given type allows "None" as a value.
    """
    can_be_none = get_origin(field.type) in [Union, UnionType] and NoneType in get_args(
        field.type
    )
    has_default_val = field.default is not dataclasses.MISSING
    has_default_factory = field.default_factory is not dataclasses.MISSING

    return can_be_none or has_default_val or has_default_factory


def _get_default(field: Field[Any]) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None
