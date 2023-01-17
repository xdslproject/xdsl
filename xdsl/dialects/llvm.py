from __future__ import annotations
from typing import TYPE_CHECKING, Annotated

from xdsl.ir import (MLIRType, ParametrizedAttribute, Attribute, Dialect,
                     Operation)
from xdsl.irdl import (Operand, ParameterDef, AnyAttr, irdl_attr_definition,
                       AttributeDef, irdl_op_definition, builder)
from xdsl.ir import OpResult, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayOfConstraint, ArrayAttr

if TYPE_CHECKING:
    from xdsl.parser import BaseParser
    from xdsl.printer import Printer


@irdl_attr_definition
class LLVMStructType(ParametrizedAttribute, MLIRType):
    name = "llvm.struct"

    # An empty string refers to a struct without a name.
    struct_name: ParameterDef[StringAttr]
    types: ParameterDef[ArrayAttr[Attribute]]

    # TODO: Add this parameter once xDSL supports the necessary capabilities.
    #  bitmask = ParameterDef(StringAttr)

    @staticmethod
    @builder
    def from_type_list(types: list[Attribute]) -> LLVMStructType:
        return LLVMStructType(
            [StringAttr.from_str(""),
             ArrayAttr.from_list(types)])

    def print_parameters(self, printer: Printer) -> None:
        assert self.struct_name.data == ""
        printer.print("<(")
        printer.print_list(self.types.data, printer.print_attribute)
        printer.print(")>")

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.must_parse_characters("<(", "LLVM Struct must start with `<(`")
        params = parser.must_parse_list_of(parser.try_parse_attribute, "Malformed LLVM struct, expected attribute definition here!")
        parser.must_parse_characters(")>", "Unexpected input, expected end of LLVM struct!")
        return [StringAttr.from_str(""), ArrayAttr.from_list(params)]


@irdl_op_definition
class LLVMExtractValue(Operation):
    name = "llvm.extractvalue"

    position = AttributeDef(ArrayOfConstraint(AnyAttr()))
    container: Annotated[Operand, AnyAttr()]

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class LLVMInsertValue(Operation):
    name = "llvm.insertvalue"

    position = AttributeDef(ArrayOfConstraint(AnyAttr()))
    container: Annotated[Operand, AnyAttr()]
    value: Annotated[Operand, AnyAttr()]

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class LLVMMLIRUndef(Operation):
    name = "llvm.mlir.undef"

    res: Annotated[OpResult, AnyAttr()]


LLVM = Dialect([LLVMExtractValue, LLVMInsertValue, LLVMMLIRUndef],
               [LLVMStructType])
