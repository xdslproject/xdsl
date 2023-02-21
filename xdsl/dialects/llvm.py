from __future__ import annotations
from typing import TYPE_CHECKING, Annotated

from xdsl.dialects.builtin import (StringAttr, ArrayAttr, DenseArrayBase,
                                   IntAttr, NoneAttr, IntegerType, IntegerAttr,
                                   AnyIntegerAttr, IndexType)
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
    addr_space: ParameterDef[IntAttr | NoneAttr]

    def print_parameters(self, printer: Printer) -> None:
        if isinstance(self.type, NoneAttr):
            return

        printer.print_string("<")
        printer.print_attribute(self.type)
        if not isinstance(self.addr_space, NoneAttr):
            printer.print_string(", ")
            printer.print_attribute(self.addr_space)

        printer.print_string(">")

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        if not parser.tokenizer.starts_with('<'):
            return [NoneAttr(), NoneAttr()]
        parser.parse_characters('<', "llvm.ptr parameters expected")
        type = parser.try_parse_type()
        if type is None:
            parser.raise_error(
                "Expected first parameter of llvm.ptr to be a type!")
        if not parser.tokenizer.starts_with(','):
            parser.parse_characters('>',
                                    "End of llvm.ptr parameters expected!")
            return [type, NoneAttr()]
        parser.parse_characters(',', "llvm.ptr args must be separated by `,`")
        addr_space = parser.parse_int_literal()
        parser.parse_characters('>', "End of llvm.ptr parameters expected!")
        return [type, IntegerAttr.from_int_and_width(addr_space, IndexType())]

    @staticmethod
    def untyped():
        return LLVMPointerType([NoneAttr(), NoneAttr()])

    @staticmethod
    def typed(type: Attribute):
        return LLVMPointerType([type, NoneAttr()])


@irdl_op_definition
class AllocaOp(Operation):
    name = "llvm.alloca"

    size: Annotated[Operand, IntegerType]

    alignment: OpAttr[AnyIntegerAttr]

    res: OpResult

    @classmethod
    def get(cls,
            size: SSAValue | Operation,
            elem_type: Attribute,
            alignment: int = 32,
            as_untyped_ptr: bool = False):
        attrs: dict[str, Attribute] = {
            'alignment': IntegerAttr.from_int_and_width(alignment, 64)
        }
        if as_untyped_ptr:
            ptr_type = LLVMPointerType.untyped()
            attrs['elem_type'] = elem_type
        else:
            ptr_type = LLVMPointerType.typed(elem_type)

        return cls.build(operands=[size],
                         attributes=attrs,
                         result_types=[ptr_type])


@irdl_op_definition
class IntToPtrOp(Operation):
    name = "llvm.inttoptr"

    input: Annotated[Operand, IntegerType]

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
class LoadOp(Operation):
    name = "llvm.load"

    ptr: Annotated[Operand, LLVMPointerType]

    dereferenced_value: Annotated[OpResult, Attribute]

    @classmethod
    def get(cls,
            ptr: SSAValue | Operation,
            result_type: Attribute | None = None):
        if result_type is None:
            ptr = SSAValue.get(ptr)
            assert isinstance(ptr.typ, LLVMPointerType)

            if isinstance(ptr.typ.type, NoneAttr):
                raise ValueError(
                    "llvm.load requires either a result type or a typed pointer!"
                )
            result_type = ptr.typ.type

        return cls.build(operands=[ptr], result_types=[result_type])


@irdl_op_definition
class NullOp(Operation):
    name = "llvm.mlir.null"

    nullptr: Annotated[OpResult, LLVMPointerType]

    @classmethod
    def get(cls, ptr_type: LLVMPointerType | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.untyped()
        assert isinstance(ptr_type, LLVMPointerType)

        return cls.build(result_types=[ptr_type])


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


LLVM = Dialect([
    LLVMExtractValue,
    LLVMInsertValue,
    LLVMMLIRUndef,
    AllocaOp,
    IntToPtrOp,
    NullOp,
    LoadOp,
], [LLVMStructType, LLVMPointerType])
