from abc import ABC, abstractmethod
from dataclasses import dataclass, Field
from types import UnionType, NoneType

from xdsl.utils.hints import isa
from xdsl.dialects import builtin
from xdsl.ir import MLContext
from typing import ClassVar, TypeVar, get_origin, Any, Union, get_args
from xdsl.utils.parse_pipeline import (
    PipelinePassSpec,
    PassArgListType,
    PassArgElementType,
)

_T = TypeVar("_T", bound="ModulePass")


@dataclass
class ModulePass(ABC):
    """
    A Pass is a named rewrite pass over an IR module.

    All passes are expected to leave the IR in a valid state after application.
    That is, the IR verifies. In turn, all passes can expect the IR they are
    applied to be in a valid state.
    """

    name: ClassVar[str]

    @abstractmethod
    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        ...

    @classmethod
    def from_pass_spec(cls: type[_T], spec: PipelinePassSpec) -> _T:
        """
        This method takes a PipelinePassSpec, does type checking on the
        arguments, and then instantiates an instance of the ModulePass
        from the spec.
        """
        # some sanity checks here
        assert issubclass(cls, ModulePass), f"{cls} must be subclass of ModulePass"
        assert hasattr(cls, "__dataclass_fields__"), f"{cls} must be a dataclass"
        assert spec.name == cls.name, "Wrong pass name provided"

        # get all dataclass fields
        fields: dict[str, Field[_T]] = cls.__dataclass_fields__

        # start constructing the argument dict for the dataclass
        arg_dict = dict[str, PassArgListType | PassArgElementType | None]()

        # iterate over all fields of the dataclass
        for field in fields.values():
            # ignore the name field and everything that's not used by __init__
            if field.name == "name" or not field.init:
                continue
            # check that non-optional fields are present
            if field.name not in spec.args and not _is_optional(field.type):
                raise Exception(f'Pass {cls.name} requires argument "{field.name}"')

            # convert pass arg to the correct type:
            arg_dict[field.name] = _convert_pass_arg_to_type(
                # we use default value [] here, this makes handling optionals easier
                spec.args.pop(field.name, []), field.type
            )
            # we use .pop here to also remove the arg from the dict

        # if not all args were removed we raise an error
        if len(spec.args) != 0:
            raise Exception(f'Unrecognised pass argument "{list(spec.args)[0]}"')

        # instantiate the dataclass using kwargs
        return cls(**arg_dict)


def _convert_pass_arg_to_type(
    value: PassArgListType, dest_type: Any
) -> PassArgListType | PassArgElementType | None:
    """
    Takes in a list of pass args, and converts them to the required type.

    value,      dest_type,      result
    []          int | None      None
    [1]         int | None      1
    [1,2]       list[int]       [1,2]
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
                raise Exception("Argument must contain a value")

    # first check if an individual value passes the type check
    if len(value) == 1 and isa(value[0], dest_type):
        return value[0]

    # then check if array value is okay
    if isa(value, dest_type):
        return value

    # at this point we exhausted all possibilities
    raise Exception(f"Incompatible types: given {value}, expected {dest_type}")


def _is_optional(field_type: Any):
    """
    Shorthand to check if the given type allows "None" as a value.
    """
    return get_origin(field_type) in [Union, UnionType] and NoneType in get_args(
        field_type
    )
