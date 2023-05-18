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
        assert issubclass(cls, ModulePass), f"{cls} must be subclass of ModulePass"
        assert hasattr(cls, "__dataclass_fields__"), f"{cls} must be a dataclass"

        assert spec.name == cls.name, "Wrong pass name provided"

        fields: dict[str, Field[_T]] = cls.__dataclass_fields__

        arg_dict = dict[str, PassArgListType | PassArgElementType | None]()

        for field in fields.values():
            if field.name == "name" or not field.init:
                continue
            if field.name not in spec.args and not _is_optional(field.type):
                raise Exception(f'Pass {cls.name} requires argument "{field.name}"')

            arg_dict[field.name] = _convert_pass_type(
                spec.args.pop(field.name, []), field.type
            )

        if len(spec.args) != 0:
            raise Exception(f'Unrecognised pass argument "{list(spec.args)[0]}"')

        return cls(**arg_dict)


def _convert_pass_type(
    value: PassArgListType, dest_type: Any
) -> PassArgListType | PassArgElementType | None:
    origin = get_origin(dest_type)

    # we need to special case optionals as [] means no option given
    if origin in [Union, UnionType]:
        if len(value) == 0:
            if NoneType in get_args(dest_type):
                return None
            else:
                raise Exception("Argument must contain a value")

    if len(value) == 1 and isa(value[0], dest_type):
        return value[0]

    if isa(value, dest_type):
        return value
    raise Exception(f"Incompatible types: given {value}, expected {dest_type}")


def _is_optional(field_type: Any):
    return get_origin(field_type) in [Union, UnionType] and NoneType in get_args(
        field_type
    )
