from __future__ import annotations

from enum import Enum
from types import EllipsisType
from typing import Literal, Sequence

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    ContainerType,
    DenseArrayBase,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    OptOperand,
    ParameterDef,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

GEP_USE_SSA_VAL = -2147483648
"""

This is used in the getelementptr index list to signify that an ssa value
should be used for this index.

"""


@irdl_attr_definition
class LLVMStructType(ParametrizedAttribute, TypeAttribute):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types
    """

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
        printer.print("<")
        if self.struct_name.data:
            printer.print_string_literal(self.struct_name.data)
            printer.print_string(", ")
        printer.print("(")
        printer.print_list(self.types.data, printer.print_attribute)
        printer.print(")>")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<", " in LLVM struct")
        struct_name = parser.parse_optional_str_literal()
        if struct_name is None:
            struct_name = ""
        else:
            parser.parse_characters(",", " after type")

        params = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_type
        )
        parser.parse_characters(">", " to close LLVM struct parameters")
        return [StringAttr(struct_name), ArrayAttr(params)]


@irdl_attr_definition
class LLVMPointerType(
    ParametrizedAttribute, TypeAttribute, ContainerType[Attribute | None]
):
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

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_characters("<") is None:
            return [NoneAttr(), NoneAttr()]
        type = parser.parse_optional_type()
        if type is None:
            parser.raise_error("Expected first parameter of llvm.ptr to be a type!")
        if parser.parse_optional_characters(",") is None:
            parser.parse_characters(">", " for llvm.ptr parameters")
            return [type, NoneAttr()]
        parser.parse_characters(",", " between llvm.ptr args")
        addr_space = parser.parse_integer()
        parser.parse_characters(">", " to end llvm.ptr parameters")
        return [type, IntegerAttr(addr_space, IndexType())]

    @staticmethod
    def opaque():
        return LLVMPointerType([NoneAttr(), NoneAttr()])

    @staticmethod
    def typed(type: Attribute):
        return LLVMPointerType([type, NoneAttr()])

    def is_typed(self):
        return not isinstance(self.type, NoneAttr)

    def get_element_type(self) -> Attribute | None:
        return self.type


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

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_characters("<") is None:
            return [NoneAttr(), NoneAttr()]
        size = IntAttr(parser.parse_integer())
        if parser.parse_optional_characters(">") is not None:
            return [size, NoneAttr()]
        parser.parse_shape_delimiter()
        type = parser.parse_optional_type()
        if type is None:
            parser.raise_error("Expected second parameter of llvm.array to be a type!")
        parser.parse_characters(">", " to end llvm.array parameters")
        return [size, type]

    @staticmethod
    def from_size_and_type(size: int | IntAttr, type: Attribute):
        if isinstance(size, int):
            size = IntAttr(size)
        return LLVMArrayType([size, type])


@irdl_attr_definition
class LLVMVoidType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.void"


@irdl_attr_definition
class LLVMFunctionType(ParametrizedAttribute, TypeAttribute):
    """
    Currently does not support variadics.

    https://mlir.llvm.org/docs/Dialects/LLVM/#function-types
    """

    name = "llvm.func"

    inputs: ParameterDef[ArrayAttr[Attribute]]
    output: ParameterDef[Attribute]
    variadic: ParameterDef[UnitAttr | NoneAttr]

    def __init__(
        self,
        inputs: Sequence[Attribute] | ArrayAttr[Attribute],
        output: Attribute | None = None,
        is_variadic: bool = False,
    ) -> None:
        if not isinstance(inputs, ArrayAttr):
            inputs = ArrayAttr(inputs)
        if output is None:
            output = LLVMVoidType()
        variad_attr = UnitAttr() if is_variadic else NoneAttr()
        super().__init__([inputs, output, variad_attr])

    @property
    def is_variadic(self) -> bool:
        return isinstance(self.variadic, UnitAttr)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        if isinstance(self.output, LLVMVoidType):
            printer.print("void")
        else:
            printer.print_attribute(self.output)

        printer.print(" (")
        printer.print_list(self.inputs, printer.print_attribute)
        if self.is_variadic:
            if self.inputs:
                printer.print(", ")
            printer.print("...")

        printer.print_string(")>")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<", " in llvm.func parameters")
        if parser.parse_optional_characters("void"):
            output = LLVMVoidType()
        else:
            output = parser.parse_attribute()

        # save pos before args for error message printing
        pos = parser.pos

        def _parse_attr_or_variadic() -> Attribute | EllipsisType:
            """
            This returns either an attribute, or Ellipsis if a
            varargs specifier (`...`) was parsed.
            """
            if parser.parse_optional_characters("...") is not None:
                return ...
            return parser.parse_attribute()

        inputs = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, _parse_attr_or_variadic
        )
        is_varargs: NoneAttr | UnitAttr = NoneAttr()
        if inputs and inputs[-1] is Ellipsis:
            is_varargs = UnitAttr()
            inputs = inputs[:-1]

        if not isa(inputs, list[Attribute]):
            parser.raise_error(
                "Varargs specifier `...` must be at the end of the argument definition",
                pos,
                parser.pos,
            )

        parser.parse_characters(">", " in llvm.func parameters")

        return [ArrayAttr(inputs), output, is_varargs]


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

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<", "llvm.linkage parameter expected")
        # The linkage string is output from xDSL as a string (and accepted by MLIR as such)
        # however it is always output from MLIR without quotes. Therefore need to determine
        # whether this is a string or not and slightly change how we parse based upon that
        linkage_str = parser.parse_optional_str_literal()
        if linkage_str is None:
            linkage_str = parser.parse_identifier()
        linkage = StringAttr(linkage_str)
        parser.parse_characters(">", " to end llvm.linkage parameters")
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
    """
    llvm.getelementptr is an instruction to do pointer arithmetic by
    adding/subtracting offsets from a pointer.

    The llvm.getelementptr is documented in various places online:

    LLVM documentation: https://www.llvm.org/docs/GetElementPtr.html
    A good blogpost: https://blog.yossarian.net/2020/09/19/LLVMs-getelementptr-by-example
    MLIR documentation: https://mlir.llvm.org/docs/Dialects/LLVM/#llvmgetelementptr-mlirllvmgepop

    Note that the first two discuss *LLVM IRs* GEP operation, not the MLIR one.
    The semantics are the same, but the structure used by MLIR is not well
    documented (yet) and the syntax is a bit different.

    Here we focus on MLIRs GEP operation:

    %res = llvm.getelementptr %ptr  [1, 2, %val]
                              ^^^^   ^^^^^^^^^^
                              input   indices

    The central point to understanding GEP is that:
    > GEP never dereferences, it only does math on the given pointer

    It *always* returns a pointer to the element "selected" that is some
    number of bytes offset from the input pointer:

    `result = ptr + x` for some x parametrized by the arguments

    ## Examples:

    Given the following pointer:

    %ptr : llvm.ptr<llvm.struct<(i32, i32, llvm.array<2xi32>)>>

    The following indices point to the following things:

    [0]      -> The first element of the pointer, so a pointer to the struct:
                llvm.ptr<llvm.struct<(i32, i32, llvm.array<2xi32>)>>

    [1]      -> The *next* element of the pointer, useful if the
                pointer points to a list of structs.
                Equivalent to (ptr + 1), so points to
                llvm.ptr<llvm.struct<(i32, i32, llvm.array<2xi32>)>>

    [0,0]    -> The first member of the first struct:
                llvm.ptr<i32>

    [1,0]    -> The first member of the *second* struct pointed to by ptr
                (can result in out-of-bounds access if the ptr only points to a single struct)
                llvm.ptr<i32>

    [0,2]    -> The third member of the first struct.
                llvm.ptr<llvm.array<2,i32>>

    [0,2,0]  -> The first entry of the array that is the third member of
                the first struct pointed to by our ptr.
                llvm.ptr<i32>

    [0,0,1]  -> Invalid! The first element of the first struct has no "sub-elements"!


    Here is an example of invalid GEP operation parameters:

    Given a different pointer to the example above:

    %ptr : llvm.ptr<llvm.struct<(llvm.ptr<i32>, i32)>>

    Note the two pointers, one to the struct, one in the struct.

    We can do math on the first pointer:

    [0]      -> First struct
                llvm.ptr<llvm.struct<(llvm.ptr<i32>, i32)>>

    [0,1]    -> Second member of first struct
                llvm.ptr<i32>

    [0,0]    -> First member of the first struct
                llvm.ptr<llvm.ptr<i32>>

    [0,0,3]  -> Invalid! In order to find the fourth element in the pointer
                it would need to be dereferenced! GEP can't do that!

    Expressed in "C", this would equate to:

    # address of first struct
    (ptr + 0)

    # address of first field of first struct
    &((ptr + 0)->elm0)
               ^^^^^^
               Even though it looks like it, we are not actually
               dereferencing ptr here.

    # address of fourth element:
    &(((ptr + 0)->elm0 + 3))
                ^^^^^^^^^^
                This actually dereferences (ptr + 0) to access elm0!

    Which translates to roughly this MLIR code:

    %elm0_addr   = llvm.gep %ptr[0,0]   : (!llvm.ptr<...>) -> !llvm.ptr<!llvm.ptr<i32>>
    %elm0        = llvm.load %elm0_addr : (!llvm.ptr<llvm.ptr<i32>>) -> !llvm.ptr<i32>
    %elm0_3_addr = llvm.gep %elm0[3]    : !llvm.ptr<i32> -> !llvm.ptr<i32>

    Here the necessary dereferencing is very visible, as %elm0_3_addr is only
    accessible through an `llvm.load` on %elm0_addr.
    """

    name = "llvm.getelementptr"

    ptr: Operand = operand_def(LLVMPointerType)
    ssa_indices: VarOperand = var_operand_def(IntegerType)
    elem_type: Attribute | None = opt_attr_def(Attribute)

    result: OpResult = result_def(LLVMPointerType)

    rawConstantIndices: DenseArrayBase = attr_def(DenseArrayBase)
    inbounds: UnitAttr | None = opt_attr_def(UnitAttr)

    @staticmethod
    def get(
        ptr: SSAValue | Operation,
        indices: Sequence[int],
        ssa_indices: Sequence[SSAValue | Operation] | None = None,
        result_type: LLVMPointerType = LLVMPointerType.opaque(),
        inbounds: bool = False,
        pointee_type: Attribute | None = None,
    ):
        """
        A basic constructor for the GEPOp.

        Pass the GEP_USE_SSA_VAL magic value in place of each constant
        index that you want to be read from an SSA value.

        Take a look at `from_mixed_indices` for something without
        magic values.
        """
        # construct default mutable argument here:
        if ssa_indices is None:
            ssa_indices = []

        # convert a potential Operation into an SSAValue
        ptr_val = SSAValue.get(ptr)
        ptr_type = ptr_val.type

        if not isinstance(ptr_type, LLVMPointerType):
            raise ValueError("Input must be a pointer")

        attrs: dict[str, Attribute] = {
            "rawConstantIndices": DenseArrayBase.create_dense_int_or_index(
                i32, indices
            ),
        }

        if not ptr_type.is_typed():
            if pointee_type is None:
                raise ValueError("Opaque types must have a pointee type passed")
            # opaque input ptr => opaque output ptr
            attrs["elem_type"] = LLVMPointerType.opaque()

        if inbounds:
            attrs["inbounds"] = UnitAttr()

        return GEPOp.build(
            operands=[ptr, ssa_indices], result_types=[result_type], attributes=attrs
        )

    @staticmethod
    def from_mixed_indices(
        ptr: SSAValue | Operation,
        indices: Sequence[int | SSAValue | Operation],
        result_type: LLVMPointerType = LLVMPointerType.opaque(),
        inbounds: bool = False,
        pointee_type: Attribute | None = None,
    ):
        """
        This is a helper function that accepts a mixed list of SSA values and const
        indices. It will automatically construct the correct indices and ssa_indices
        lists from that.

        You can call this using [1, 2, some_ssa_val, 3] as the indices array.

        Other than that, this behaves exactly the same as `.get`
        """
        ssa_indices: list[SSAValue] = []
        const_indices: list[int] = []
        for idx in indices:
            if isinstance(idx, int):
                const_indices.append(idx)
            else:
                const_indices.append(GEP_USE_SSA_VAL)
                ssa_indices.append(SSAValue.get(idx))
        return GEPOp.get(
            ptr,
            const_indices,
            ssa_indices,
            result_type=result_type,
            inbounds=inbounds,
            pointee_type=pointee_type,
        )


@irdl_op_definition
class AllocaOp(IRDLOperation):
    name = "llvm.alloca"

    size: Operand = operand_def(IntegerType)

    alignment: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    res: OpResult = result_def()

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

    input: Operand = operand_def(IntegerType)

    output: OpResult = result_def(LLVMPointerType)

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

    input: Operand = operand_def(LLVMPointerType)

    output: OpResult = result_def(IntegerType)

    @staticmethod
    def get(arg: SSAValue | Operation, int_type: Attribute = i64):
        return PtrToIntOp.build(operands=[arg], result_types=[int_type])


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "llvm.load"

    ptr: Operand = operand_def(LLVMPointerType)

    dereferenced_value: OpResult = result_def()

    @staticmethod
    def get(ptr: SSAValue | Operation, result_type: Attribute | None = None):
        if result_type is None:
            ptr = SSAValue.get(ptr)
            assert isinstance(ptr.type, LLVMPointerType)

            if isinstance(ptr.type.type, NoneAttr):
                raise ValueError(
                    "llvm.load requires either a result type or a typed pointer!"
                )
            result_type = ptr.type.type

        return LoadOp.build(operands=[ptr], result_types=[result_type])


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "llvm.store"

    value: Operand = operand_def()
    ptr: Operand = operand_def(LLVMPointerType)

    alignment: IntegerAttr[IntegerType] | None = opt_attr_def(IntegerAttr[IntegerType])
    ordering: IntegerAttr[IntegerType] | None = opt_attr_def(IntegerAttr[IntegerType])
    volatile_: UnitAttr | None = opt_attr_def(UnitAttr)
    nontemporal: UnitAttr | None = opt_attr_def(UnitAttr)

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

    nullptr: OpResult = result_def(LLVMPointerType)

    @staticmethod
    def get(ptr_type: LLVMPointerType | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.opaque()
        assert isinstance(ptr_type, LLVMPointerType)

        return NullOp.build(result_types=[ptr_type])


@irdl_op_definition
class ExtractValueOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmextractvalue-mlirllvmextractvalueop
    """

    name = "llvm.extractvalue"

    position: DenseArrayBase = attr_def(DenseArrayBase)
    container: Operand = operand_def(Attribute)

    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        position: DenseArrayBase,
        container: SSAValue | Operation,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[container],
            attributes={
                "position": position,
            },
            result_types=[result_type],
        )


@irdl_op_definition
class InsertValueOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvminsertvalue-mlirllvminsertvalueop
    """

    name = "llvm.insertvalue"

    position: DenseArrayBase = attr_def(DenseArrayBase)
    container: Operand = operand_def(Attribute)
    value: Operand = operand_def(Attribute)

    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        position: DenseArrayBase,
        container: SSAValue,
        value: SSAValue,
    ):
        super().__init__(
            operands=[container, value],
            attributes={
                "position": position,
            },
            result_types=[container.type],
        )


@irdl_op_definition
class UndefOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirundef-mlirllvmundefop
    """

    name = "llvm.mlir.undef"

    res: OpResult = result_def(Attribute)

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])


@irdl_op_definition
class GlobalOp(IRDLOperation):
    name = "llvm.mlir.global"

    global_type: Attribute = attr_def(Attribute)
    constant: UnitAttr | None = opt_attr_def(UnitAttr)
    sym_name: StringAttr = attr_def(StringAttr)
    linkage: LinkageAttr = attr_def(LinkageAttr)
    dso_local: UnitAttr | None = opt_attr_def(UnitAttr)
    thread_local_: UnitAttr | None = opt_attr_def(UnitAttr)
    value: Attribute | None = opt_attr_def(Attribute)
    alignment: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)
    addr_space: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    unnamed_addr: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)
    section: StringAttr | None = opt_attr_def(StringAttr)

    # This always needs an empty region as it is in the top level module definition
    body: Region = region_def()

    @staticmethod
    def get(
        global_type: Attribute,
        sym_name: str | StringAttr,
        linkage: str | LinkageAttr,
        addr_space: int = 0,
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

        return GlobalOp.build(attributes=attrs, regions=[Region([])])


@irdl_op_definition
class AddressOfOp(IRDLOperation):
    name = "llvm.mlir.addressof"

    global_name: SymbolRefAttr = attr_def(SymbolRefAttr)
    result: OpResult = result_def(LLVMPointerType)

    @staticmethod
    def get(
        global_name: str | StringAttr | SymbolRefAttr, result_type: LLVMPointerType
    ):
        if isinstance(global_name, (StringAttr, str)):
            global_name = SymbolRefAttr(global_name)

        return AddressOfOp.build(
            attributes={"global_name": global_name}, result_types=[result_type]
        )


LLVM_CALLING_CONVS: set[str] = {
    "ccc",
    "fastcc",
    "coldcc",
    "cc 10",
    "cc 11",
    "webkit_jscc",
    "anyregcc",
    "preserve_mostcc",
    "preserve_allcc",
    "cxx_fast_tlscc",
    "tailcc",
    "swiftcc",
    "swifttailcc",
    "cfguard_checkcc",
}
"""
A list of all valid calling conventions understood by LLVM, see
https://llvm.org/docs/LangRef.html#calling-conventions
for more info.
"""


@irdl_attr_definition
class CallingConventionAttr(ParametrizedAttribute):
    """
    LLVM Calling convention, default is ccc.
    """

    name = "llvm.cconv"

    convention: ParameterDef[StringAttr]

    @property
    def cconv_name(self) -> str:
        return self.convention.data

    def __init__(self, conv: str):
        super().__init__([StringAttr(conv)])

    def _verify(self):
        if self.cconv_name not in LLVM_CALLING_CONVS:
            raise VerifyException(f'Invalid calling convention "{self.cconv_name}"')

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<" + self.convention.data + ">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        for conv in LLVM_CALLING_CONVS:
            if parser.parse_optional_characters(conv) is not None:
                parser.parse_characters(">")
                return [StringAttr(conv)]
        parser.raise_error(f"Unknown calling convention")


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "llvm.func"

    body: Region = region_def()
    sym_name: StringAttr = attr_def(StringAttr)
    function_type: LLVMFunctionType = attr_def(LLVMFunctionType)
    CConv: CallingConventionAttr = attr_def(CallingConventionAttr)
    linkage: LinkageAttr = attr_def(LinkageAttr)
    visibility_: IntegerAttr[IntegerType] = attr_def(IntegerAttr[IntegerType])

    def __init__(
        self,
        sym_name: str | StringAttr,
        function_type: LLVMFunctionType,
        linkage: LinkageAttr = LinkageAttr("internal"),
        cconv: CallingConventionAttr = CallingConventionAttr("ccc"),
        visibility: int | IntegerAttr[IntegerType] = 0,
        body: Region | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(visibility, int):
            visibility = IntegerAttr.from_int_and_width(visibility, 64)
        if body is None:
            body = Region([])
        super().__init__(
            operands=[],
            regions=[body],
            attributes={
                "sym_name": sym_name,
                "function_type": function_type,
                "CConv": cconv,
                "linkage": linkage,
                "visibility_": visibility,
            },
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmreturn-mlirllvmreturnop
    """

    name = "llvm.return"

    arg: OptOperand = opt_operand_def(Attribute)

    traits = frozenset((IsTerminator(),))

    def __init__(self, value: Attribute | None = None):
        super().__init__(attributes={"value": value})


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "llvm.mlir.constant"
    result: OpResult = result_def(Attribute)
    value: Attribute = attr_def(Attribute)

    def __init__(self, value: Attribute, value_type: Attribute):
        super().__init__(attributes={"value": value}, result_types=[value_type])


@irdl_op_definition
class CallIntrinsicOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcall_intrinsic-mlirllvmcallintrinsicop
    """

    name = "llvm.call_intrinsic"

    intrin: StringAttr = attr_def(StringAttr)
    args: VarOperand = var_operand_def()
    ress: VarOpResult = var_result_def()

    def __init__(
        self,
        intrin: StringAttr | str,
        args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
    ):
        if isinstance(intrin, str):
            intrin = StringAttr(intrin)
        super().__init__(
            operands=args,
            result_types=(result_types,),
            attributes={
                "intrin": intrin,
            },
        )


class FastMathFlag(Enum):
    REASSOC = "reassoc"
    NO_NANS = "nnan"
    NO_INFS = "ninf"
    NO_SIGNED_ZEROS = "nsz"
    ALLOW_RECIP = "arcp"
    ALLOW_CONTRACT = "contract"
    APPROX_FUNC = "afn"

    @staticmethod
    def try_parse(parser: AttrParser) -> set[FastMathFlag] | None:
        if parser.parse_optional_characters("none") is not None:
            return set[FastMathFlag]()
        if parser.parse_optional_characters("fast") is not None:
            return set(FastMathFlag)

        for option in FastMathFlag:
            if parser.parse_optional_characters(option.value) is not None:
                return {option}

        return None


@irdl_attr_definition
class FastMathAttr(Data[tuple[FastMathFlag, ...]]):
    name = "llvm.fastmath"

    @property
    def flags(self) -> set[FastMathFlag]:
        """
        Returns a copy of the fast math flags.
        """
        return set(self.data)

    def __init__(self, flags: None | Sequence[FastMathFlag] | Literal["none", "fast"]):
        flags_: set[FastMathFlag]
        match flags:
            case "none" | None:
                flags_ = set()
            case "fast":
                flags_ = set(FastMathFlag)
            case other:
                flags_ = set(other)

        super().__init__(tuple(flags_))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[FastMathFlag, ...]:
        flags = FastMathFlag.try_parse(parser)
        if flags is None:
            return tuple()

        while parser.parse_optional_punctuation(",") is not None:
            flag = parser.expect(
                lambda: FastMathFlag.try_parse(parser), "fastmath flag expected"
            )
            flags.update(flag)

        return tuple(flags)

    def print_parameter(self, printer: Printer):
        flags = self.data
        if len(flags) == 0:
            printer.print("none")
        elif len(flags) == len(FastMathFlag):
            printer.print("fast")
        else:
            # make sure we emit flags in a consistent order
            printer.print(
                ",".join(flag.value for flag in FastMathFlag if flag in flags)
            )


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "llvm.call"

    args: VarOperand = var_operand_def()

    callee: SymbolRefAttr = attr_def(SymbolRefAttr)
    fastmathFlags: FastMathAttr = attr_def(FastMathAttr)

    def __init__(
        self,
        callee: str | SymbolRefAttr | StringAttr,
        *args: SSAValue | Operation,
        fastmath: FastMathAttr = FastMathAttr(None),
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        super().__init__(
            operands=[args],
            attributes={
                "callee": callee,
                "fastmathFlags": fastmath,
            },
        )


LLVM = Dialect(
    [
        ExtractValueOp,
        InsertValueOp,
        UndefOp,
        AllocaOp,
        GEPOp,
        IntToPtrOp,
        NullOp,
        LoadOp,
        StoreOp,
        GlobalOp,
        AddressOfOp,
        FuncOp,
        CallOp,
        ReturnOp,
        ConstantOp,
        CallIntrinsicOp,
    ],
    [
        LLVMStructType,
        LLVMPointerType,
        LLVMArrayType,
        LLVMVoidType,
        LLVMFunctionType,
        LinkageAttr,
        CallingConventionAttr,
        FastMathAttr,
    ],
)
