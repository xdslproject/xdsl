from __future__ import annotations
from typing import TYPE_CHECKING, Annotated

from xdsl.dialects.builtin import (StringAttr, ArrayAttr, DenseArrayBase,
                                   IntAttr, NoneAttr, IntegerType,
                                   AnyIntegerAttr, IntegerAttr)
from xdsl.ir import (MLIRType, ParametrizedAttribute, Attribute, Dialect,
                     OpResult, Operation, SSAValue)
from xdsl.irdl import (OpAttr, Operand, ParameterDef, AnyAttr,
                       irdl_attr_definition, irdl_op_definition)

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
        parser.parse_characters("<(", "LLVM Struct must start with `<(`")
        params = parser.parse_list_of(
            parser.try_parse_type,
            "Malformed LLVM struct, expected attribute definition here!")
        parser.parse_characters(
            ")>", "Unexpected input, expected end of LLVM struct!")
        return [StringAttr.from_str(""), ArrayAttr.from_list(params)]


@irdl_attr_definition
class LLVMPointerType(ParametrizedAttribute, MLIRType):
    name = "llvm.ptr"

    type: ParameterDef[Attribute | NoneAttr]
    target: ParameterDef[IntAttr | NoneAttr]

    def print_parameters(self, printer: Printer) -> None:
        if isinstance(self.type, NoneAttr):
            return

        printer.print_string("<")
        printer.print_attribute(self.type)
        if not isinstance(self.target, NoneAttr):
            printer.print_string(", ")
            printer.print_attribute(self.target)

        printer.print_string(">")

    @classmethod
    def untyped(cls):
        return cls([NoneAttr(), NoneAttr()])

    @classmethod
    def typed(cls, type: Attribute):
        return cls([type, NoneAttr()])


@irdl_op_definition
class AllocaOp(Operation):
    name = "llvm.alloca"

    size: Annotated[Operand, AnyIntegerAttr]

    alignment: OpAttr[IntegerType.from_width(64)]
    #elem_type: OpAttr[Attribute]

    res: Annotated[OpResult, LLVMPointerType]

    @classmethod
    def get(cls,
            size: SSAValue | Operation,
            elem_type: Attribute,
            alignment: int = 32):
        return cls.build(
            operands=[size],
            attributes={
                'alignment': IntegerAttr.from_int_and_width(alignment, 64)
            },
            result_types=[LLVMPointerType([elem_type, NoneAttr()])])


@irdl_op_definition
class IntToPtrOp(Operation):
    name = "llvm.inttoptr"

    input: Annotated[Operand, AnyIntegerAttr]

    output: Annotated[OpResult, LLVMPointerType]

    @classmethod
    def get(cls,
            input: SSAValue | Operation,
            ptr_type: Attribute | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.untyped()
        else:
            ptr_type = LLVMPointerType.typed(ptr_type)
        return cls.build(operands=[input], result_types=[ptr_type])


@irdl_op_definition
class LLVMExtractValue(Operation):
    name = "llvm.extractvalue"

    position: OpAttr[DenseArrayBase]
    container: Annotated[Operand, AnyAttr()]

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class LLVMInsertValue(Operation):
    name = "llvm.insertvalue"

    position: OpAttr[DenseArrayBase]
    container: Annotated[Operand, AnyAttr()]
    value: Annotated[Operand, AnyAttr()]

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class LLVMMLIRUndef(Operation):
    name = "llvm.mlir.undef"

    res: Annotated[OpResult, AnyAttr()]


LLVM = Dialect(
    [LLVMExtractValue, LLVMInsertValue, LLVMMLIRUndef, AllocaOp, IntToPtrOp],
    [LLVMStructType, LLVMPointerType])
