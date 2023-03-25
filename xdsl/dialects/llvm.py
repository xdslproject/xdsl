from __future__ import annotations
from typing import TYPE_CHECKING, Annotated, Sequence

from xdsl.dialects.builtin import (StringAttr, ArrayAttr, DenseArrayBase,
                                   IntAttr, NoneAttr, IntegerType, IntegerAttr,
                                   AnyIntegerAttr, IndexType, UnitAttr, i32,
                                   i64)
from xdsl.ir import (MLIRType, ParametrizedAttribute, Attribute, Dialect,
                     OpResult, Operation, SSAValue)
from xdsl.irdl import (OpAttr, Operand, ParameterDef, AnyAttr,
                       irdl_attr_definition, irdl_op_definition, VarOperand,
                       OptOpAttr)

if TYPE_CHECKING:
    from xdsl.parser import BaseParser
    from xdsl.printer import Printer

GEP_USE_SSA_VAL = -2147483648
"""

This is used in the getelementptr index list to signify that an ssa value 
should be used for this index.

"""


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
        return LLVMStructType([StringAttr(""), ArrayAttr(types)])

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
        return [StringAttr(""), ArrayAttr(params)]


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
        return [type, IntegerAttr.from_params(addr_space, IndexType())]

    @staticmethod
    def opaque():
        return LLVMPointerType([NoneAttr(), NoneAttr()])

    @staticmethod
    def typed(type: Attribute):
        return LLVMPointerType([type, NoneAttr()])

    def is_typed(self):
        return not isinstance(self.type, NoneAttr)


@irdl_op_definition
class GEPOp(Operation):
    """
    The getelementptr is rather difficult to understand. There's a lot of
    good literature online on how this works exactly. Here's my attempt
    at explaining how it works and doesn't work using a few examples:

    But first, you need to always keep in mind that:
    > GEP never dereferences, it only does math on the given pointer

    It *always* returns a pointer to the element you "selected"!

    ## Examples:

    Given an ptr:

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
                (can result in oob access if the ptr only points to a single struct)
                llvm.ptr<i32>

    [0,2]    -> The third member of the first struct.
                llvm.ptr<llvm.array<2,i32>>

    [0,2,0]  -> The first entry of the array that is the third member of
                the first struct pointed to by our ptr.
                llvm.ptr<i32>

    [0,0,1]  -> Invalid! The first element of the first struct has no "sub-elements"!


    Remember that GEP NEVER DEREFERENCES! So given a pointer like this:

    %ptr : llvm.ptr<llvm.struct<(llvm.ptr<i32>, i32)>>

    We can do math on the first pointer:

    [0]      -> First struct
                llvm.ptr<llvm.struct<(llvm.ptr<i32>, i32)>>

    [0,1]    -> Second member of first struct
                llvm.ptr<i32>

    [0,0]    -> First member of the first struct
                llvm.ptr<llvm.ptr<i32>>

    [0,0,3]  -> Invalid! In order to find the fourth element in the pointer
                we would need to dereference it! GEP can't do that!

    In "C" you'd need to do the following:

    # address of first struct
    (ptr + 0)
    # address of first field of first struct
    &((ptr + 0)->elm0)
    # address of fourth element:
    &(((ptr + 0)->elm0 + 3))
                ^^^^^^^^^^
    Which translates to roughly this code:
    %elm0_addr   = llvm.gep %ptr[0,0]  : (!llvm.ptr<...>) -> !llvm.ptr<!llvm.ptr<i32>>
    %elm0        = llvm.load %elm0_adr : (!llvm.ptr<llvm.ptr<i32>>) -> !llvm.ptr<i32>
    %elm0_3_addr = llvm.gep %elm0[3]   : !llvm.ptr<i32> -> !llvm.ptr<i32>

    As you see, you can only get the %elm0_3_addr by dereferencing the %elem_0_addr once!

    I hope this clears things up!
    """
    name = "llvm.getelementptr"

    ptr: Annotated[Operand, LLVMPointerType]
    ssa_indices: Annotated[VarOperand, IntegerType]
    elem_type: OptOpAttr[Attribute]

    result: Annotated[OpResult, LLVMPointerType]

    rawConstantIndices: OpAttr[DenseArrayBase]
    inbounds: OptOpAttr[UnitAttr]

    @staticmethod
    def get(ptr: SSAValue | Operation,
            indices: Sequence[int],
            ssa_indices: Sequence[SSAValue | Operation] | None = None,
            result_type: LLVMPointerType = LLVMPointerType.opaque(),
            inbounds: bool = False,
            pointee_type: Attribute | None = None):
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
        ptr_type = ptr_val.typ

        if not isinstance(result_type, LLVMPointerType):
            raise ValueError('Result type must be a pointer.')

        if not isinstance(ptr_type, LLVMPointerType):
            raise ValueError('Input must be a pointer')

        attrs: dict[str, Attribute] = {
            'rawConstantIndices':
            DenseArrayBase.create_dense_int_or_index(i32, indices),
        }

        if not ptr_type.is_typed():
            if pointee_type is None:
                raise ValueError(
                    'Opaque types must have a pointee type passed')
            # opaque input ptr => opaque output ptr
            attrs['elem_type'] = LLVMPointerType.opaque()

        if inbounds:
            attrs['inbounds'] = UnitAttr()

        return GEPOp.build(operands=[ptr, ssa_indices],
                           result_types=[result_type],
                           attributes=attrs)

    @staticmethod
    def from_mixed_indices(
            ptr: SSAValue | Operation,
            indices: list[int | SSAValue | Operation],
            result_type: LLVMPointerType = LLVMPointerType.opaque(),
            inbounds: bool = False,
            pointee_type: Attribute | None = None):
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
        return GEPOp.get(ptr,
                         const_indices,
                         ssa_indices,
                         result_type=result_type,
                         inbounds=inbounds,
                         pointee_type=pointee_type)


@irdl_op_definition
class AllocaOp(Operation):
    name = "llvm.alloca"

    size: Annotated[Operand, IntegerType]

    alignment: OpAttr[AnyIntegerAttr]

    res: OpResult

    @staticmethod
    def get(size: SSAValue | Operation,
            elem_type: Attribute,
            alignment: int = 32,
            as_untyped_ptr: bool = False):
        attrs: dict[str, Attribute] = {
            'alignment': IntegerAttr.from_int_and_width(alignment, 64)
        }
        if as_untyped_ptr:
            ptr_type = LLVMPointerType.opaque()
            attrs['elem_type'] = elem_type
        else:
            ptr_type = LLVMPointerType.typed(elem_type)

        return AllocaOp.build(operands=[size],
                              attributes=attrs,
                              result_types=[ptr_type])


@irdl_op_definition
class IntToPtrOp(Operation):
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
class PtrToIntOp(Operation):
    name = "llvm.ptrtoint"

    input: Annotated[Operand, LLVMPointerType]

    output: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(arg: SSAValue | Operation, int_type: Attribute = i64):
        return PtrToIntOp.build(operands=[arg], result_types=[int_type])


@irdl_op_definition
class LoadOp(Operation):
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
class StoreOp(Operation):
    name = "llvm.store"

    value: Operand
    ptr: Annotated[Operand, LLVMPointerType]

    alignment: OptOpAttr[IntegerAttr[IntegerType]]
    ordering: OptOpAttr[IntegerAttr[IntegerType]]
    volatile_: OptOpAttr[UnitAttr]
    nontemporal: OptOpAttr[UnitAttr]

    @staticmethod
    def get(value: SSAValue | Operation,
            ptr: SSAValue | Operation,
            alignment: int | None = None,
            ordering: int = 0,
            volatile: bool = False,
            nontemporal: bool = False):
        attrs: dict[str, Attribute] = {
            'ordering': IntegerAttr(ordering, i64),
        }

        if alignment is not None:
            attrs['alignment'] = IntegerAttr[IntegerType](alignment, i64)
        if volatile:
            attrs['volatile_'] = UnitAttr()
        if nontemporal:
            attrs['nontemporal'] = UnitAttr()

        return StoreOp.build(
            operands=[value, ptr],
            attributes=attrs,
            result_types=[],
        )


@irdl_op_definition
class NullOp(Operation):
    name = "llvm.mlir.null"

    nullptr: Annotated[OpResult, LLVMPointerType]

    @classmethod
    def get(cls, ptr_type: LLVMPointerType | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.opaque()
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
    GEPOp,
    IntToPtrOp,
    NullOp,
    LoadOp,
    StoreOp,
], [LLVMStructType, LLVMPointerType])
