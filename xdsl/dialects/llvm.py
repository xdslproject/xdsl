from __future__ import annotations
from typing import TYPE_CHECKING, Annotated, Sequence

from xdsl.dialects.builtin import (
    StringAttr,
    ArrayAttr,
    DenseArrayBase,
    IntAttr,
    NoneAttr,
    IntegerType,
    IntegerAttr,
    AnyIntegerAttr,
    IndexType,
    UnitAttr,
    i32,
    i64,
    SymbolRefAttr,
)
from xdsl.ir import (
    Block,
    TypeAttribute,
    ParametrizedAttribute,
    Attribute,
    Dialect,
    OpResult,
    Operation,
    SSAValue,
    Region,
)
from xdsl.irdl import (
    OpAttr,
    Operand,
    ParameterDef,
    AnyAttr,
    irdl_attr_definition,
    irdl_op_definition,
    VarOperand,
    OptOpAttr,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.parser import Parser
    from xdsl.printer import Printer

GEP_USE_SSA_VAL = -2147483648
"""

This is used in the getelementptr index list to signify that an ssa value
should be used for this index.

"""


@irdl_attr_definition
class LLVMStructType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.struct"

    # An empty string refers to a struct without a name.
    struct_name: ParameterDef[StringAttr]
    types: ParameterDef[ArrayAttr[Attribute]]

    # TODO: Add this parameter once xDSL supports the necessary capabilities.
    #  bitmask = ParameterDef(StringAttr)

    @staticmethod
    def from_type_list(types: Sequence[Attribute]) -> LLVMStructType:
        return LLVMStructType([StringAttr(""), ArrayAttr(types)])

    def print_parameters(self, printer: Printer) -> None:
        assert self.struct_name.data == ""
        printer.print("<(")
        printer.print_list(self.types.data, printer.print_attribute)
        printer.print(")>")

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_characters("<(", "LLVM Struct must start with `<(`")
        params = parser.parse_list_of(
            parser.try_parse_type,
            "Malformed LLVM struct, expected attribute definition here!",
        )
        parser.parse_characters(")>", "Unexpected input, expected end of LLVM struct!")
        return [StringAttr(""), ArrayAttr(params)]


@irdl_attr_definition
class LLVMPointerType(ParametrizedAttribute, TypeAttribute):
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
    def parse_parameters(parser: Parser) -> list[Attribute]:
        if not parser.tokenizer.starts_with("<"):
            return [NoneAttr(), NoneAttr()]
        parser.parse_characters("<", "llvm.ptr parameters expected")
        type = parser.try_parse_type()
        if type is None:
            parser.raise_error("Expected first parameter of llvm.ptr to be a type!")
        if not parser.tokenizer.starts_with(","):
            parser.parse_characters(">", "End of llvm.ptr parameters expected!")
            return [type, NoneAttr()]
        parser.parse_characters(",", "llvm.ptr args must be separated by `,`")
        addr_space = parser.parse_int_literal()
        parser.parse_characters(">", "End of llvm.ptr parameters expected!")
        return [type, IntegerAttr.from_params(addr_space, IndexType())]

    @staticmethod
    def opaque():
        return LLVMPointerType([NoneAttr(), NoneAttr()])

    @staticmethod
    def typed(type: Attribute):
        return LLVMPointerType([type, NoneAttr()])

    def is_typed(self):
        return not isinstance(self.type, NoneAttr)


@irdl_attr_definition
class LLVMArrayType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.array"

    size: ParameterDef[IntAttr]
    type: ParameterDef[Attribute]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_string(str(self.size.data))
        printer.print_string(" x ")
        printer.print_attribute(self.type)
        printer.print_string(">")

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        if not parser.tokenizer.starts_with("<"):
            return [NoneAttr(), NoneAttr()]
        parser.parse_characters("<", "llvm.array parameters expected")
        size = IntAttr(parser.parse_int_literal())
        if not parser.tokenizer.starts_with("x"):
            parser.parse_characters(">", "End of llvm.array type expected!")
            return [size, NoneAttr()]
        parser.parse_characters(
            "x", "llvm.array size and type must be separated by `x`"
        )
        type = parser.try_parse_type()
        if type is None:
            parser.raise_error("Expected second parameter of llvm.array to be a type!")
        parser.parse_characters(">", "End of llvm.array parameters expected!")
        return [size, type]

    @staticmethod
    def from_size_and_type(size: int | IntAttr, type: Attribute):
        if isinstance(size, int):
            size = IntAttr(size)
        return LLVMArrayType([size, type])


@irdl_attr_definition
class LinkageAttr(ParametrizedAttribute):
    name = "llvm.linkage"

    linkage: ParameterDef[StringAttr]

    def __init__(self, linkage: str | StringAttr) -> None:
        if isinstance(linkage, str):
            linkage = StringAttr(linkage)
        super().__init__([linkage])

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.linkage)
        printer.print_string(">")

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_characters("<", "llvm.linkage parameter expected")
        # The linkage string is output from xDSL as a string (and accepted by MLIR as such)
        # however it is always output from MLIR without quotes. Therefore need to determine
        # whether this is a string or not and slightly change how we parse based upon that
        linkage_str = parser.try_parse_string_literal()
        if linkage_str is not None:
            linkage_str = linkage_str.string_contents
        else:
            linkage_str = parser.tokenizer.next_token().text
        linkage = StringAttr(linkage_str)
        parser.parse_characters(">", "End of llvm.linkage parameter expected!")
        return [linkage]

    def verify(self):
        allowed_linkage = [
            "private",
            "internal",
            "available_externally",
            "linkonce",
            "weak",
            "common",
            "appending",
            "extern_weak",
            "linkonce_odr",
            "weak_odr",
            "external",
        ]
        if self.linkage.data not in allowed_linkage:
            raise VerifyException(f"Specified linkage '{self.linkage.data}' is unknown")


@irdl_op_definition
class GEPOp(IRDLOperation):
    name = "llvm.getelementptr"

    ptr: Annotated[Operand, LLVMPointerType]
    ssa_indices: Annotated[VarOperand, IntegerType]
    elem_type: OptOpAttr[Attribute]
    rawConstantIndices: OpAttr[DenseArrayBase]
    inbounds: OptOpAttr[UnitAttr]
    result: Annotated[OpResult, LLVMPointerType]

    @staticmethod
    def get(
        ptr: SSAValue | Operation,
        result_type: LLVMPointerType = LLVMPointerType.opaque(),
        indices: list[int]
        | None = None,  # Here we are assuming the indices follow the MLIR standard (min int where the SSA value should be used)
        ssa_indices: list[SSAValue | Operation] | None = None,
        inbounds: bool = False,
        pointee_type: Attribute | None = None,
    ):
        if indices is None:
            raise ValueError("llvm.getelementptr must have indices passed.")

        indices_attr = DenseArrayBase.create_dense_int_or_index(i32, indices)

        # construct default mutable argument here:
        if ssa_indices is None:
            ssa_indices = []

        # convert a potential Operation into an SSAValue
        ptr_val = SSAValue.get(ptr)
        ptr_type = ptr_val.typ

        if not isinstance(result_type, LLVMPointerType):
            raise ValueError("Result type must be a pointer.")

        if not isinstance(ptr_type, LLVMPointerType):
            raise ValueError("Input must be a pointer")

        if not ptr_type.is_typed():
            if pointee_type == None:
                raise ValueError("Opaque types must have a pointee type passed")

        attrs: dict[str, Attribute] = {
            "rawConstantIndices": indices_attr,
        }

        if not ptr_type.is_typed():
            attrs["elem_type"] = result_type

        if inbounds:
            attrs["inbounds"] = UnitAttr()

        return GEPOp.build(
            operands=[ptr, ssa_indices], result_types=[result_type], attributes=attrs
        )


@irdl_op_definition
class AllocaOp(IRDLOperation):
    name = "llvm.alloca"

    size: Annotated[Operand, IntegerType]

    alignment: OpAttr[AnyIntegerAttr]

    res: OpResult

    @staticmethod
    def get(
        size: SSAValue | Operation,
        elem_type: Attribute,
        alignment: int = 32,
        as_untyped_ptr: bool = False,
    ):
        attrs: dict[str, Attribute] = {
            "alignment": IntegerAttr.from_int_and_width(alignment, 64)
        }
        if as_untyped_ptr:
            ptr_type = LLVMPointerType.opaque()
            attrs["elem_type"] = elem_type
        else:
            ptr_type = LLVMPointerType.typed(elem_type)

        return AllocaOp.build(
            operands=[size], attributes=attrs, result_types=[ptr_type]
        )


@irdl_op_definition
class IntToPtrOp(IRDLOperation):
    name = "llvm.inttoptr"

    input: Annotated[Operand, IntegerType]

    output: Annotated[OpResult, LLVMPointerType]

    @staticmethod
    def get(input: SSAValue | Operation, ptr_type: Attribute | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.opaque()
        else:
            ptr_type = LLVMPointerType.typed(ptr_type)
        return IntToPtrOp.build(operands=[input], result_types=[ptr_type])


@irdl_op_definition
class PtrToIntOp(IRDLOperation):
    name = "llvm.ptrtoint"

    input: Annotated[Operand, LLVMPointerType]

    output: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(arg: SSAValue | Operation, int_type: Attribute = i64):
        return PtrToIntOp.build(operands=[arg], result_types=[int_type])


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "llvm.load"

    ptr: Annotated[Operand, LLVMPointerType]

    dereferenced_value: OpResult

    @staticmethod
    def get(ptr: SSAValue | Operation, result_type: Attribute | None = None):
        if result_type is None:
            ptr = SSAValue.get(ptr)
            assert isinstance(ptr.typ, LLVMPointerType)

            if isinstance(ptr.typ.type, NoneAttr):
                raise ValueError(
                    "llvm.load requires either a result type or a typed pointer!"
                )
            result_type = ptr.typ.type

        return LoadOp.build(operands=[ptr], result_types=[result_type])


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "llvm.store"

    value: Operand
    ptr: Annotated[Operand, LLVMPointerType]

    alignment: OptOpAttr[IntegerAttr[IntegerType]]
    ordering: OptOpAttr[IntegerAttr[IntegerType]]
    volatile_: OptOpAttr[UnitAttr]
    nontemporal: OptOpAttr[UnitAttr]

    @staticmethod
    def get(
        value: SSAValue | Operation,
        ptr: SSAValue | Operation,
        alignment: int | None = None,
        ordering: int = 0,
        volatile: bool = False,
        nontemporal: bool = False,
    ):
        attrs: dict[str, Attribute] = {
            "ordering": IntegerAttr(ordering, i64),
        }

        if alignment is not None:
            attrs["alignment"] = IntegerAttr[IntegerType](alignment, i64)
        if volatile:
            attrs["volatile_"] = UnitAttr()
        if nontemporal:
            attrs["nontemporal"] = UnitAttr()

        return StoreOp.build(
            operands=[value, ptr],
            attributes=attrs,
            result_types=[],
        )


@irdl_op_definition
class NullOp(IRDLOperation):
    name = "llvm.mlir.null"

    nullptr: Annotated[OpResult, LLVMPointerType]

    @staticmethod
    def get(ptr_type: LLVMPointerType | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.opaque()
        assert isinstance(ptr_type, LLVMPointerType)

        return NullOp.build(result_types=[ptr_type])


@irdl_op_definition
class LLVMExtractValue(IRDLOperation):
    name = "llvm.extractvalue"

    position: OpAttr[DenseArrayBase]
    container: Annotated[Operand, AnyAttr()]

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class LLVMInsertValue(IRDLOperation):
    name = "llvm.insertvalue"

    position: OpAttr[DenseArrayBase]
    container: Annotated[Operand, AnyAttr()]
    value: Annotated[Operand, AnyAttr()]

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class LLVMMLIRUndef(IRDLOperation):
    name = "llvm.mlir.undef"

    res: Annotated[OpResult, AnyAttr()]


@irdl_op_definition
class GlobalOp(IRDLOperation):
    name = "llvm.mlir.global"

    global_type: OpAttr[Attribute]
    constant: OptOpAttr[UnitAttr]
    sym_name: OpAttr[StringAttr]
    linkage: OpAttr[LinkageAttr]
    dso_local: OptOpAttr[UnitAttr]
    thread_local_: OptOpAttr[UnitAttr]
    value: OptOpAttr[Attribute]
    alignment: OptOpAttr[AnyIntegerAttr]
    addr_space: OpAttr[AnyIntegerAttr]
    unnamed_addr: OptOpAttr[AnyIntegerAttr]
    section: OptOpAttr[StringAttr]

    # This always needs an empty region as it is in the top level module definition
    body: Region

    @staticmethod
    def get(
        global_type: Attribute,
        sym_name: str | StringAttr,
        linkage: str | LinkageAttr,
        addr_space: int,
        constant: bool | None = None,
        dso_local: bool | None = None,
        thread_local_: bool | None = None,
        value: Attribute | None = None,
        alignment: int | None = None,
        unnamed_addr: int | None = None,
        section: str | StringAttr | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)

        if isinstance(linkage, str):
            linkage = LinkageAttr(linkage)

        attrs: dict[str, Attribute] = {
            "global_type": global_type,
            "sym_name": sym_name,
            "linkage": linkage,
            "addr_space": IntegerAttr(addr_space, 32),
        }

        if constant is not None and constant:
            attrs["constant"] = UnitAttr()

        if dso_local is not None and dso_local:
            attrs["dso_local"] = UnitAttr()

        if thread_local_ is not None and thread_local_:
            attrs["thread_local_"] = UnitAttr()

        if value is not None:
            attrs["value"] = value

        if alignment is not None:
            attrs["alignment"] = IntegerAttr(alignment, 64)

        if unnamed_addr is not None:
            attrs["unnamed_addr"] = IntegerAttr(unnamed_addr, 64)

        if section is not None:
            if isinstance(section, str):
                section = StringAttr(section)
            attrs["section"] = section

        return GlobalOp.build(attributes=attrs, regions=[Region([Block()])])


@irdl_op_definition
class AddressOfOp(IRDLOperation):
    name = "llvm.mlir.addressof"

    global_name: OpAttr[SymbolRefAttr]
    result: Annotated[OpResult, LLVMPointerType]

    @staticmethod
    def get(
        global_name: str | StringAttr | SymbolRefAttr, result_type: LLVMPointerType
    ):
        if isinstance(global_name, str):
            global_name = StringAttr(global_name)
        if isinstance(global_name, StringAttr):
            global_name = SymbolRefAttr(global_name)

        return AddressOfOp.build(
            attributes={"global_name": global_name}, result_types=[result_type]
        )


LLVM = Dialect(
    [
        LLVMExtractValue,
        LLVMInsertValue,
        LLVMMLIRUndef,
        AllocaOp,
        GEPOp,
        IntToPtrOp,
        NullOp,
        LoadOp,
        StoreOp,
        GlobalOp,
        AddressOfOp,
    ],
    [LLVMStructType, LLVMPointerType, LLVMArrayType, LinkageAttr],
)
