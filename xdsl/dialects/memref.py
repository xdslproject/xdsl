from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Iterable,
    Sequence,
    TypeVar,
    Optional,
    List,
    TypeAlias,
    cast,
)

from xdsl.utils.hints import isa
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    DenseArrayBase,
    IndexType,
    StridedLayoutAttr,
    ArrayAttr,
    NoneAttr,
    SymbolRefAttr,
    i64,
    StringAttr,
    UnitAttr,
    i32,
    IntegerType,
)
from xdsl.ir import (
    TypeAttribute,
    Operation,
    SSAValue,
    ParametrizedAttribute,
    Dialect,
    OpResult,
)
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    Generic,
    Attribute,
    AnyAttr,
    Operand,
    VarOperand,
    AttrSizedOperandSegments,
    OpAttr,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

if TYPE_CHECKING:
    from xdsl.parser import Parser
    from xdsl.printer import Printer

_MemRefTypeElement = TypeVar("_MemRefTypeElement", bound=Attribute)


@irdl_attr_definition
class MemRefType(Generic[_MemRefTypeElement], ParametrizedAttribute, TypeAttribute):
    name = "memref"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_MemRefTypeElement]
    layout: ParameterDef[Attribute]
    memory_space: ParameterDef[Attribute]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    @staticmethod
    def from_element_type_and_shape(
        referenced_type: _MemRefTypeElement,
        shape: Iterable[int | AnyIntegerAttr],
        layout: Attribute = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> MemRefType[_MemRefTypeElement]:
        return MemRefType(
            [
                ArrayAttr[AnyIntegerAttr](
                    [
                        d
                        if isinstance(d, IntegerAttr)
                        else IntegerAttr.from_index_int_value(d)
                        for d in shape
                    ]
                ),
                referenced_type,
                layout,
                memory_space,
            ]
        )

    @staticmethod
    def from_params(
        referenced_type: _MemRefTypeElement,
        shape: ArrayAttr[AnyIntegerAttr] = ArrayAttr(
            [IntegerAttr.from_int_and_width(1, 64)]
        ),
        layout: Attribute = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> MemRefType[_MemRefTypeElement]:
        return MemRefType([shape, referenced_type, layout, memory_space])

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser._synchronize_lexer_and_tokenizer()  # pyright: ignore[reportPrivateUsage]
        parser.parse_punctuation("<", " in memref attribute")
        shape = parser.parse_attribute()
        parser.parse_punctuation(",", " between shape and element type parameters")
        type = parser.parse_attribute()
        # If we have a layout or a memory space, parse both of them.
        if parser.parse_optional_punctuation(",") is None:
            parser.parse_punctuation(">", " at end of memref attribute")
            return [shape, type, NoneAttr(), NoneAttr()]
        layout = parser.parse_attribute()
        parser.parse_punctuation(",", " between layout and memory space")
        memory_space = parser.parse_attribute()
        parser.parse_punctuation(">", " at end of memref attribute")
        parser._synchronize_lexer_and_tokenizer()  # pyright: ignore[reportPrivateUsage]

        return [shape, type, layout, memory_space]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.shape, ", ", self.element_type)
        if self.layout != NoneAttr() or self.memory_space != NoneAttr():
            printer.print(", ", self.layout, ", ", self.memory_space)
        printer.print(">")


_UnrankedMemrefTypeElems = TypeVar(
    "_UnrankedMemrefTypeElems", bound=Attribute, covariant=True
)
_UnrankedMemrefTypeElemsInit = TypeVar("_UnrankedMemrefTypeElemsInit", bound=Attribute)


@irdl_attr_definition
class UnrankedMemrefType(
    Generic[_UnrankedMemrefTypeElems], ParametrizedAttribute, TypeAttribute
):
    name = "unranked_memref"

    element_type: ParameterDef[_UnrankedMemrefTypeElems]
    memory_space: ParameterDef[Attribute]

    @staticmethod
    def from_type(
        referenced_type: _UnrankedMemrefTypeElemsInit,
        memory_space: Attribute = NoneAttr(),
    ) -> UnrankedMemrefType[_UnrankedMemrefTypeElemsInit]:
        return UnrankedMemrefType([referenced_type, memory_space])


AnyUnrankedMemrefType: TypeAlias = UnrankedMemrefType[Attribute]


@irdl_op_definition
class Load(IRDLOperation):
    name = "memref.load"
    memref: Annotated[Operand, MemRefType[Attribute]]
    indices: Annotated[VarOperand, IndexType]
    res: Annotated[OpResult, AnyAttr()]

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters,
    # which is subject to change

    def verify_(self):
        if not isinstance(self.memref.typ, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_typ = cast(MemRefType[Attribute], self.memref.typ)

        if memref_typ.element_type != self.res.typ:
            raise Exception("expected return type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @staticmethod
    def get(ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]) -> Load:
        ssa_value = SSAValue.get(ref)
        typ = ssa_value.typ
        typ = cast(MemRefType[Attribute], typ)
        return Load.build(operands=[ref, indices], result_types=[typ.element_type])


@irdl_op_definition
class Store(IRDLOperation):
    name = "memref.store"
    value: Annotated[Operand, AnyAttr()]
    memref: Annotated[Operand, MemRefType[Attribute]]
    indices: Annotated[VarOperand, IndexType]

    def verify_(self):
        if not isinstance(self.memref.typ, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_typ = cast(MemRefType[Attribute], self.memref.typ)

        if memref_typ.element_type != self.value.typ:
            raise Exception("Expected value type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("Expected an index for each dimension")

    @staticmethod
    def get(
        value: Operation | SSAValue,
        ref: Operation | SSAValue,
        indices: Sequence[Operation | SSAValue],
    ) -> Store:
        return Store.build(operands=[value, ref, indices])


@irdl_op_definition
class Alloc(IRDLOperation):
    name = "memref.alloc"

    dynamic_sizes: Annotated[VarOperand, IndexType]
    symbol_operands: Annotated[VarOperand, IndexType]

    memref: Annotated[OpResult, MemRefType[Attribute]]

    # TODO how to constraint the IntegerAttr type?
    alignment: OpAttr[AnyIntegerAttr]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        return_type: Attribute,
        alignment: int,
        shape: Optional[Iterable[int | AnyIntegerAttr]] = None,
    ) -> Alloc:
        if shape is None:
            shape = [1]
        return Alloc.build(
            operands=[[], []],
            result_types=[MemRefType.from_element_type_and_shape(return_type, shape)],
            attributes={"alignment": IntegerAttr.from_int_and_width(alignment, 64)},
        )


@irdl_op_definition
class Alloca(IRDLOperation):
    name = "memref.alloca"

    dynamic_sizes: Annotated[VarOperand, IndexType]
    symbol_operands: Annotated[VarOperand, IndexType]

    memref: Annotated[OpResult, MemRefType[Attribute]]

    # TODO how to constraint the IntegerAttr type?
    alignment: OpAttr[AnyIntegerAttr]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        return_type: Attribute,
        alignment: int,
        shape: Optional[Iterable[int | AnyIntegerAttr]] = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
    ) -> Alloca:
        if shape is None:
            shape = [1]

        if dynamic_sizes is None:
            dynamic_sizes = []

        return Alloca.build(
            operands=[dynamic_sizes, []],
            result_types=[MemRefType.from_element_type_and_shape(return_type, shape)],
            attributes={"alignment": IntegerAttr.from_int_and_width(alignment, 64)},
        )


@irdl_op_definition
class Dealloc(IRDLOperation):
    name = "memref.dealloc"
    memref: Annotated[Operand, MemRefType[Attribute] | UnrankedMemrefType[Attribute]]

    @staticmethod
    def get(operand: Operation | SSAValue) -> Dealloc:
        return Dealloc.build(operands=[operand])


@irdl_op_definition
class GetGlobal(IRDLOperation):
    name = "memref.get_global"
    memref: Annotated[OpResult, MemRefType[Attribute]]

    def verify_(self) -> None:
        if "name" not in self.attributes:
            raise VerifyException("GetGlobal requires a 'name' attribute")

        if not isinstance(self.attributes["name"], SymbolRefAttr):
            raise VerifyException("expected 'name' attribute to be a SymbolRefAttr")

    @staticmethod
    def get(name: str, return_type: Attribute) -> GetGlobal:
        return GetGlobal.build(
            result_types=[return_type], attributes={"name": SymbolRefAttr(name)}
        )

    # TODO how to verify the types, as the global might be defined in another
    # compilation unit


@irdl_op_definition
class Global(IRDLOperation):
    name = "memref.global"

    sym_name: OpAttr[StringAttr]
    sym_visibility: OpAttr[StringAttr]
    type: OpAttr[Attribute]
    initial_value: OpAttr[Attribute]

    def verify_(self) -> None:
        if not isinstance(self.type, MemRefType):
            raise Exception("Global expects a MemRefType")

        if not isinstance(self.initial_value, UnitAttr | DenseIntOrFPElementsAttr):
            raise Exception(
                "Global initial value is expected to be a "
                "dense type or an unit attribute"
            )

    @staticmethod
    def get(
        sym_name: StringAttr,
        typ: Attribute,
        initial_value: Attribute,
        sym_visibility: StringAttr = StringAttr("private"),
    ) -> Global:
        return Global.build(
            attributes={
                "sym_name": sym_name,
                "type": typ,
                "initial_value": initial_value,
                "sym_visibility": sym_visibility,
            }
        )


@irdl_op_definition
class Dim(IRDLOperation):
    name = "memref.dim"

    source: Annotated[Operand, MemRefType[Attribute] | UnrankedMemrefType[Attribute]]
    index: Annotated[Operand, IndexType]

    result: Annotated[OpResult, IndexType]

    @staticmethod
    def from_source_and_index(
        source: SSAValue | Operation, index: SSAValue | Operation
    ):
        return Dim.build(operands=[source, index], result_types=[IndexType()])


@irdl_op_definition
class Rank(IRDLOperation):
    name = "memref.rank"

    source: Annotated[Operand, MemRefType[Attribute]]

    rank: Annotated[OpResult, IndexType]

    @staticmethod
    def from_memref(memref: Operation | SSAValue):
        return Rank.build(operands=[memref], result_types=[IndexType()])


@irdl_op_definition
class ExtractAlignedPointerAsIndexOp(IRDLOperation):
    name = "memref.extract_aligned_pointer_as_index"

    source: Annotated[Operand, MemRefType]

    aligned_pointer: Annotated[OpResult, IndexType]

    @staticmethod
    def get(source: SSAValue | Operation):
        return ExtractAlignedPointerAsIndexOp.build(
            operands=[source], result_types=[IndexType()]
        )


@irdl_op_definition
class Subview(IRDLOperation):
    name = "memref.subview"

    source: Annotated[Operand, MemRefType]
    offsets: Annotated[VarOperand, IndexType]
    sizes: Annotated[VarOperand, IndexType]
    strides: Annotated[VarOperand, IndexType]
    static_offsets: OpAttr[DenseArrayBase]
    static_sizes: OpAttr[DenseArrayBase]
    static_strides: OpAttr[DenseArrayBase]
    result: Annotated[OpResult, MemRefType]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def from_static_parameters(
        source: SSAValue | Operation,
        source_element_type: Attribute,
        source_shape: Sequence[int],
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int],
    ) -> Subview:
        source = SSAValue.get(source)

        layout_strides = [1]
        for input_size in reversed(source_shape[1:]):
            layout_strides.insert(0, layout_strides[0] * input_size)

        layout_offset = sum(
            stride * offset for stride, offset in zip(layout_strides, offsets)
        )

        for i in range(len(layout_strides)):
            layout_strides[i] *= strides[i]

        layout = StridedLayoutAttr(layout_strides, layout_offset)

        return_typ = MemRefType.from_element_type_and_shape(
            source_element_type,
            sizes,
            layout,
        )

        return Subview.build(
            operands=[source, [], [], []],
            result_types=[return_typ],
            attributes={
                "static_offsets": DenseArrayBase.from_list(i64, offsets),
                "static_sizes": DenseArrayBase.from_list(i64, sizes),
                "static_strides": DenseArrayBase.from_list(i64, strides),
            },
        )


@irdl_op_definition
class Cast(IRDLOperation):
    name = "memref.cast"

    source: Annotated[Operand, MemRefType | UnrankedMemrefType]
    dest: Annotated[OpResult, MemRefType | UnrankedMemrefType]

    @staticmethod
    def get(
        source: SSAValue | Operation,
        type: MemRefType[Attribute] | UnrankedMemrefType[Attribute],
    ):
        return Cast.build(operands=[source], result_types=[type])


@irdl_op_definition
class DmaStartOp(IRDLOperation):
    name = "memref.dma_start"

    src: Annotated[Operand, MemRefType]
    src_indices: Annotated[VarOperand, IndexType]

    dest: Annotated[Operand, MemRefType]
    dest_indices: Annotated[VarOperand, IndexType]

    num_elements: Annotated[Operand, IndexType]

    tag: Annotated[Operand, MemRefType[IntegerType]]
    tag_indices: Annotated[VarOperand, IndexType]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        src: SSAValue | Operation,
        src_indices: Sequence[SSAValue | Operation],
        dest: SSAValue | Operation,
        dest_indices: Sequence[SSAValue | Operation],
        num_elements: SSAValue | Operation,
        tag: SSAValue | Operation,
        tag_indices: Sequence[SSAValue | Operation],
    ):
        return DmaStartOp.build(
            operands=[
                src,
                src_indices,
                dest,
                dest_indices,
                num_elements,
                tag,
                tag_indices,
            ]
        )

    def verify_(self) -> None:
        assert isa(self.src.typ, MemRefType[Attribute])
        assert isa(self.dest.typ, MemRefType[Attribute])
        assert isa(self.tag.typ, MemRefType[IntegerType])

        if len(self.src.typ.shape) != len(self.src_indices):
            raise VerifyException(
                "Expected {} source indices (because of shape of src memref)".format(
                    len(self.src.typ.shape)
                )
            )

        if len(self.dest.typ.shape) != len(self.dest_indices):
            raise VerifyException(
                "Expected {} dest indices (because of shape of dest memref)".format(
                    len(self.dest.typ.shape)
                )
            )

        if len(self.tag.typ.shape) != len(self.tag_indices):
            raise VerifyException(
                "Expected {} tag indices (because of shape of tag memref)".format(
                    len(self.tag.typ.shape)
                )
            )

        if self.tag.typ.element_type != i32:
            raise VerifyException("Expected tag to be a memref of i32")

        if self.dest.typ.memory_space == self.src.typ.memory_space:
            raise VerifyException("Source and dest must have different memory spaces!")


@irdl_op_definition
class DmaWaitOp(IRDLOperation):
    name = "memref.dma_wait"

    tag: Annotated[Operand, MemRefType]
    tag_indices: Annotated[VarOperand, IndexType]

    num_elements: Annotated[Operand, IndexType]

    @staticmethod
    def get(
        tag: SSAValue | Operation,
        tag_indices: Sequence[SSAValue | Operation],
        num_elements: SSAValue | Operation,
    ):
        return DmaWaitOp.build(
            operands=[
                tag,
                tag_indices,
                num_elements,
            ]
        )

    def verify_(self) -> None:
        assert isa(self.tag.typ, MemRefType[Attribute])

        if len(self.tag.typ.shape) != len(self.tag_indices):
            raise VerifyException(
                f"Expected {len(self.tag.typ.shape)} tag indices because of shape of tag memref"
            )

        if self.tag.typ.element_type != i32:
            raise VerifyException("Expected tag to be a memref of i32")


MemRef = Dialect(
    [
        Load,
        Store,
        Alloc,
        Alloca,
        Dealloc,
        GetGlobal,
        Global,
        Dim,
        ExtractAlignedPointerAsIndexOp,
        Subview,
        Cast,
        DmaStartOp,
        DmaWaitOp,
    ],
    [MemRefType, UnrankedMemrefType],
)
