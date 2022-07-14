from __future__ import annotations
from dataclasses import dataclass

from xdsl.irdl import ParameterDef, AnyAttr, irdl_op_builder, irdl_attr_definition, AttributeDef, OperandDef, ResultDef, irdl_op_definition
from xdsl.ir import MLContext, ParametrizedAttribute, TYPE_CHECKING, Attribute, Operation
from typing import overload
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


@irdl_attr_definition
class LLVMStructType(ParametrizedAttribute):
    name = "llvm.struct"

    struct_name: ParameterDef[StringAttr]
    types: ParameterDef[ArrayAttr[AnyAttr()]]

    # bitmask = ParameterDef(StringAttr)

    def from_type_list(types: list[Attribute]) -> LLVMStructType:
        return LLVMStructType(
            [StringAttr.from_str(""),
             ArrayAttr.from_list(types)])


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
