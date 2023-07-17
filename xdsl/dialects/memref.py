from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Iterable, Sequence, TypeAlias, TypeVar, cast

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    ContainerType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    ShapedType,
    StridedLayoutAttr,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

if TYPE_CHECKING:
    from xdsl.parser import AttrParser
    from xdsl.printer import Printer

_MemRefTypeElement = TypeVar("_MemRefTypeElement", bound=Attribute)


@irdl_attr_definition
class MemRefType(
    Generic[_MemRefTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[_MemRefTypeElement],
):
    name = "memref"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_MemRefTypeElement]
    layout: ParameterDef[StridedLayoutAttr | NoneAttr]
    memory_space: ParameterDef[Attribute]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.value.data for i in self.shape.data)

    def get_element_type(self) -> _MemRefTypeElement:
        return self.element_type

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

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
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
    memref: Operand = operand_def(MemRefType[Attribute])
    indices: VarOperand = var_operand_def(IndexType)
    res: OpResult = result_def(AnyAttr())

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters,
    # which is subject to change

    def verify_(self):
        if not isinstance(self.memref.type, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_type = cast(MemRefType[Attribute], self.memref.type)

        if memref_type.element_type != self.res.type:
            raise Exception("expected return type to match the MemRef element type")

        if self.memref.type.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @staticmethod
    def get(ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]) -> Load:
        ssa_value = SSAValue.get(ref)
        ssa_value_type = ssa_value.type
        ssa_value_type = cast(MemRefType[Attribute], ssa_value_type)
        return Load.build(
            operands=[ref, indices], result_types=[ssa_value_type.element_type]
        )


@irdl_op_definition
class Store(IRDLOperation):
    name = "memref.store"
    value: Operand = operand_def(AnyAttr())
    memref: Operand = operand_def(MemRefType[Attribute])
    indices: VarOperand = var_operand_def(IndexType)

    def verify_(self):
        if not isinstance(self.memref.type, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_type = cast(MemRefType[Attribute], self.memref.type)

        if memref_type.element_type != self.value.type:
            raise Exception("Expected value type to match the MemRef element type")

        if self.memref.type.get_num_dims() != len(self.indices):
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

    dynamic_sizes: VarOperand = var_operand_def(IndexType)
    symbol_operands: VarOperand = var_operand_def(IndexType)

    memref: OpResult = result_def(MemRefType[Attribute])

    # TODO how to constraint the IntegerAttr type?
    alignment: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        return_type: Attribute,
        alignment: int | None = None,
        shape: Iterable[int | AnyIntegerAttr] | None = None,
    ) -> Alloc:
        if shape is None:
            shape = [1]
        return Alloc.build(
            operands=[[], []],
            result_types=[MemRefType.from_element_type_and_shape(return_type, shape)],
            attributes={
                "alignment": IntegerAttr.from_int_and_width(alignment, 64)
                if alignment is not None
                else None
            },
        )


@irdl_op_definition
class Alloca(IRDLOperation):
    name = "memref.alloca"

    dynamic_sizes: VarOperand = var_operand_def(IndexType)
    symbol_operands: VarOperand = var_operand_def(IndexType)

    memref: OpResult = result_def(MemRefType[Attribute])

    # TODO how to constraint the IntegerAttr type?
    alignment: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        return_type: Attribute,
        alignment: int | AnyIntegerAttr | None = None,
        shape: Iterable[int | AnyIntegerAttr] | None = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
    ) -> Alloca:
        if shape is None:
            shape = [1]

        if dynamic_sizes is None:
            dynamic_sizes = []

        if isinstance(alignment, int):
            alignment = IntegerAttr.from_int_and_width(alignment, 64)

        return Alloca.build(
            operands=[dynamic_sizes, []],
            result_types=[MemRefType.from_element_type_and_shape(return_type, shape)],
            attributes={
                "alignment": alignment,
            },
        )


@irdl_op_definition
class Dealloc(IRDLOperation):
    name = "memref.dealloc"
    memref: Operand = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    @staticmethod
    def get(operand: Operation | SSAValue) -> Dealloc:
        return Dealloc.build(operands=[operand])


@irdl_op_definition
class GetGlobal(IRDLOperation):
    name = "memref.get_global"
    memref: OpResult = result_def(MemRefType[Attribute])
    name_: SymbolRefAttr = attr_def(SymbolRefAttr, attr_name="name")

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

    sym_name: StringAttr = attr_def(StringAttr)
    sym_visibility: StringAttr = attr_def(StringAttr)
    type: Attribute = attr_def(Attribute)
    initial_value: Attribute = attr_def(Attribute)

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
        sym_type: Attribute,
        initial_value: Attribute,
        sym_visibility: StringAttr = StringAttr("private"),
    ) -> Global:
        return Global.build(
            attributes={
                "sym_name": sym_name,
                "type": sym_type,
                "initial_value": initial_value,
                "sym_visibility": sym_visibility,
            }
        )


@irdl_op_definition
class Dim(IRDLOperation):
    name = "memref.dim"

    source: Operand = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])
    index: Operand = operand_def(IndexType)

    result: OpResult = result_def(IndexType)

    @staticmethod
    def from_source_and_index(
        source: SSAValue | Operation, index: SSAValue | Operation
    ):
        return Dim.build(operands=[source, index], result_types=[IndexType()])


@irdl_op_definition
class Rank(IRDLOperation):
    name = "memref.rank"

    source: Operand = operand_def(MemRefType[Attribute])

    rank: OpResult = result_def(IndexType)

    @staticmethod
    def from_memref(memref: Operation | SSAValue):
        return Rank.build(operands=[memref], result_types=[IndexType()])


@irdl_op_definition
class ExtractAlignedPointerAsIndexOp(IRDLOperation):
    name = "memref.extract_aligned_pointer_as_index"

    source: Operand = operand_def(MemRefType)

    aligned_pointer: OpResult = result_def(IndexType)

    @staticmethod
    def get(source: SSAValue | Operation):
        return ExtractAlignedPointerAsIndexOp.build(
            operands=[source], result_types=[IndexType()]
        )


@irdl_op_definition
class Subview(IRDLOperation):
    name = "memref.subview"

    source: Operand = operand_def(MemRefType)
    offsets: VarOperand = var_operand_def(IndexType)
    sizes: VarOperand = var_operand_def(IndexType)
    strides: VarOperand = var_operand_def(IndexType)
    static_offsets: DenseArrayBase = attr_def(DenseArrayBase)
    static_sizes: DenseArrayBase = attr_def(DenseArrayBase)
    static_strides: DenseArrayBase = attr_def(DenseArrayBase)
    result: OpResult = result_def(MemRefType)

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def from_static_parameters(
        source: SSAValue | Operation,
        source_type: MemRefType[Attribute],
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int],
        reduce_rank: bool = False,
    ) -> Subview:
        source = SSAValue.get(source)

        source_shape = [e.value.data for e in source_type.shape.data]
        source_offset = 0
        source_strides = [1]
        for input_size in reversed(source_shape[1:]):
            source_strides.insert(0, source_strides[0] * input_size)
        if isinstance(source_type.layout, StridedLayoutAttr):
            if isinstance(source_type.layout.offset, IntAttr):
                source_offset = source_type.layout.offset.data
            if isa(source_type.layout.strides, ArrayAttr[IntAttr]):
                source_strides = [s.data for s in source_type.layout.strides]

        layout_strides = [a * b for (a, b) in zip(strides, source_strides)]

        layout_offset = (
            sum(stride * offset for stride, offset in zip(source_strides, offsets))
            + source_offset
        )

        if reduce_rank:
            composed_strides = layout_strides
            layout_strides: list[int] = []
            result_sizes: list[int] = []

            for stride, size in zip(composed_strides, sizes):
                if size == 1:
                    continue
                layout_strides.append(stride)
                result_sizes.append(size)

        else:
            result_sizes = list(sizes)

        layout = StridedLayoutAttr(layout_strides, layout_offset)

        return_type = MemRefType.from_element_type_and_shape(
            source_type.element_type,
            result_sizes,
            layout,
            source_type.memory_space,
        )

        return Subview.build(
            operands=[source, [], [], []],
            result_types=[return_type],
            attributes={
                "static_offsets": DenseArrayBase.from_list(i64, offsets),
                "static_sizes": DenseArrayBase.from_list(i64, sizes),
                "static_strides": DenseArrayBase.from_list(i64, strides),
            },
        )


@irdl_op_definition
class Cast(IRDLOperation):
    name = "memref.cast"

    source: Operand = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])
    dest: OpResult = result_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    @staticmethod
    def get(
        source: SSAValue | Operation,
        type: MemRefType[Attribute] | UnrankedMemrefType[Attribute],
    ):
        return Cast.build(operands=[source], result_types=[type])


@irdl_op_definition
class DmaStartOp(IRDLOperation):
    name = "memref.dma_start"

    src: Operand = operand_def(MemRefType)
    src_indices: VarOperand = var_operand_def(IndexType)

    dest: Operand = operand_def(MemRefType)
    dest_indices: VarOperand = var_operand_def(IndexType)

    num_elements: Operand = operand_def(IndexType)

    tag: Operand = operand_def(MemRefType[IntegerType])
    tag_indices: VarOperand = var_operand_def(IndexType)

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
        assert isa(self.src.type, MemRefType[Attribute])
        assert isa(self.dest.type, MemRefType[Attribute])
        assert isa(self.tag.type, MemRefType[IntegerType])

        if len(self.src.type.shape) != len(self.src_indices):
            raise VerifyException(
                "Expected {} source indices (because of shape of src memref)".format(
                    len(self.src.type.shape)
                )
            )

        if len(self.dest.type.shape) != len(self.dest_indices):
            raise VerifyException(
                "Expected {} dest indices (because of shape of dest memref)".format(
                    len(self.dest.type.shape)
                )
            )

        if len(self.tag.type.shape) != len(self.tag_indices):
            raise VerifyException(
                "Expected {} tag indices (because of shape of tag memref)".format(
                    len(self.tag.type.shape)
                )
            )

        if self.tag.type.element_type != i32:
            raise VerifyException("Expected tag to be a memref of i32")

        if self.dest.type.memory_space == self.src.type.memory_space:
            raise VerifyException("Source and dest must have different memory spaces!")


@irdl_op_definition
class DmaWaitOp(IRDLOperation):
    name = "memref.dma_wait"

    tag: Operand = operand_def(MemRefType)
    tag_indices: VarOperand = var_operand_def(IndexType)

    num_elements: Operand = operand_def(IndexType)

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
        assert isa(self.tag.type, MemRefType[Attribute])

        if len(self.tag.type.shape) != len(self.tag_indices):
            raise VerifyException(
                f"Expected {len(self.tag.type.shape)} tag indices because of shape of tag memref"
            )

        if self.tag.type.element_type != i32:
            raise VerifyException("Expected tag to be a memref of i32")


@irdl_op_definition
class CopyOp(IRDLOperation):
    name = "memref.copy"
    source: Operand = operand_def(MemRefType)
    destination: Operand = operand_def(MemRefType)

    def __init__(self, source: SSAValue | Operation, destination: SSAValue | Operation):
        super().__init__([source, destination])

    def verify_(self) -> None:
        source = cast(MemRefType[Attribute], self.source.type)
        destination = cast(MemRefType[Attribute], self.destination.type)
        if source.get_shape() != destination.get_shape():
            raise VerifyException(
                f"Expected source and destination to have the same shape."
            )
        if source.get_element_type() != destination.get_element_type():
            raise VerifyException(
                f"Expected source and destination to have the same element type."
            )


MemRef = Dialect(
    [
        Load,
        Store,
        Alloc,
        Alloca,
        CopyOp,
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
