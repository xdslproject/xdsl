from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xdsl.ir import (MLContext, MLIRType, ParametrizedAttribute, Attribute,
                     Operation)
from xdsl.irdl import (ParameterDef, AnyAttr, irdl_attr_definition,
                       AttributeDef, OperandDef, ResultDef, irdl_op_definition,
                       builder)
from xdsl.dialects.builtin import StringAttr, ArrayOfConstraint, ArrayAttr

if TYPE_CHECKING:
    from xdsl.parser import Parser
    from xdsl.printer import Printer


@dataclass
class LLVM:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(LLVMStructType)

        self.ctx.register_op(LLVMExtractValue)
        self.ctx.register_op(LLVMInsertValue)
        self.ctx.register_op(LLVMMLIRUndef)


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
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_string("<(")
        params = parser.parse_list(parser.parse_optional_attribute)
        parser.parse_string(")>")
        return [StringAttr.from_str(""), ArrayAttr.from_list(params)]


@irdl_op_definition
class LLVMExtractValue(Operation):
    name = "llvm.extractvalue"

    position = AttributeDef(ArrayOfConstraint(AnyAttr()))
    container = OperandDef(AnyAttr())

    res = ResultDef(AnyAttr())


@irdl_op_definition
class LLVMInsertValue(Operation):
    name = "llvm.insertvalue"

    position = AttributeDef(ArrayOfConstraint(AnyAttr()))
    container = OperandDef(AnyAttr())
    value = OperandDef(AnyAttr())

    res = ResultDef(AnyAttr())


@irdl_op_definition
class LLVMMLIRUndef(Operation):
    name = "llvm.mlir.undef"

    res = ResultDef(AnyAttr())
