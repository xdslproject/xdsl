import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import Field, dataclass, field
from types import NoneType, UnionType
from typing import Any, ClassVar, TypeVar, Union, get_args, get_origin

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.utils.hints import isa, type_repr
from xdsl.utils.parse_pipeline import (
    PassArgElementType,
    PassArgListType,
    PipelinePassSpec,
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
    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None: ...

    @classmethod
    def from_pass_spec(cls: type[ModulePassT], spec: PipelinePassSpec) -> ModulePassT:
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

        # iterate over all fields of the dataclass
        for op_field in fields:
            # ignore the name field and everything that's not used by __init__
            if op_field.name == "name" or not op_field.init:
                continue
            # check that non-optional fields are present
            if op_field.name not in spec_arguments_dict:
                if _is_optional(op_field):
                    arg_dict[op_field.name] = _get_default(op_field)
                    continue
                raise ValueError(f'Pass {cls.name} requires argument "{op_field.name}"')

            # convert pass arg to the correct type:
            arg_dict[op_field.name] = _convert_pass_arg_to_type(
                spec_arguments_dict.pop(op_field.name),
                op_field.type,
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

    def pipeline_pass_spec(self) -> PipelinePassSpec:
        """
        This function takes a ModulePass and returns a PipelinePassSpec.
        """
        # get all dataclass fields
        fields: tuple[Field[Any], ...] = dataclasses.fields(self)
        arg_dict: dict[str, PassArgListType] = {}

        # iterate over all fields of the dataclass
        for op_field in fields:
            # ignore the name field and everything that's not used by __init__
            if op_field.name == "name" or not op_field.init:
                continue

            if _is_optional(op_field):
                arg_dict[op_field.name] = _get_default(op_field)

            val = getattr(self, op_field.name)
            if val is None:
                arg_dict.update({op_field.name: ()})
            elif isinstance(val, PassArgElementType):
                arg_dict.update({op_field.name: (getattr(self, op_field.name),)})
            else:
                arg_dict.update({op_field.name: getattr(self, op_field.name)})

        return PipelinePassSpec(self.name, arg_dict)


# Git Issue: https://github.com/xdslproject/xdsl/issues/1845
def get_pass_argument_names_and_types(arg: type[ModulePassT]) -> str:
    """
    This method takes a type[ModulePassT] and outputs a string containing the names of the
    pass arguments and their types. If an argument has a default value, it is not
    added to the string.
    """

    return " ".join(
        [
            (
                f"{field.name}={type_repr(field.type)}"
                if not hasattr(arg, field.name)
                else f"{field.name}={str(getattr(arg, field.name)).lower()}"
            )
            for field in dataclasses.fields(arg)
        ]
    )


def _empty_callback(
    previous_pass: ModulePass, module: builtin.ModuleOp, next_pass: ModulePass
) -> None:
    return


@dataclass(frozen=True)
class PipelinePass(ModulePass):
    passes: tuple[ModulePass, ...]
    callback: Callable[[ModulePass, builtin.ModuleOp, ModulePass], None] = field(
        default=_empty_callback
    )
    """
    Function called in between every pass, taking the pass that just ran, the module, and
    the next pass.
    """

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        if not self.passes:
            # Early exit to avoid fetching a non-existing last pass.
            return

        for prev, next in zip(self.passes[:-1], self.passes[1:]):
            prev.apply(ctx, op)
            self.callback(prev, op, next)

        self.passes[-1].apply(ctx, op)

    @classmethod
    def build_pipeline_tuples(
        cls,
        available_passes: dict[str, Callable[[], type[ModulePass]]],
        pass_spec_pipeline: Iterable[PipelinePassSpec],
    ) -> Iterator[tuple[type[ModulePass], PipelinePassSpec]]:
        for p in pass_spec_pipeline:
            if p.name not in available_passes:
                raise Exception(f"Unrecognized pass: {p.name}")
            yield (available_passes[p.name](), p)


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
