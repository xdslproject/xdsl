from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Annotated, cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    StridedLayoutAttr,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    UnrankedMemrefType,
    i32,
    i64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParsePropInAttrDict,
    VarOperand,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalisationPatternsTrait,
    HasParent,
    IsTerminator,
    SymbolOpInterface,
)
from xdsl.utils.bitwise_casts import is_power_of_two
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_op_definition
class Load(IRDLOperation):
    T = Annotated[Attribute, ConstraintVar("T")]

    name = "memref.load"

    nontemporal = opt_prop_def(BoolAttr)

    memref: Operand = operand_def(MemRefType[T])
    indices: VarOperand = var_operand_def(IndexType())
    res: OpResult = result_def(T)

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = "$memref `[` $indices `]` attr-dict `:` type($memref)"

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters,
    # which is subject to change

    def verify_(self):
        memref_type = self.memref.type
        if not isinstance(memref_type, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_type = cast(MemRefType[Attribute], memref_type)

        if memref_type.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @classmethod
    def get(
        cls, ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]
    ) -> Self:
        ssa_value = SSAValue.get(ref)
        ssa_value_type = ssa_value.type
        ssa_value_type = cast(MemRefType[Attribute], ssa_value_type)
        return cls(operands=[ref, indices], result_types=[ssa_value_type.element_type])


@irdl_op_definition
class Store(IRDLOperation):
    T = Annotated[Attribute, ConstraintVar("T")]

    name = "memref.store"

    nontemporal = opt_prop_def(BoolAttr)

    value: Operand = operand_def(T)
    memref: Operand = operand_def(MemRefType[T])
    indices: VarOperand = var_operand_def(IndexType())

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref)"

    def verify_(self):
        if not isinstance(memref_type := self.memref.type, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_type = cast(MemRefType[Attribute], memref_type)

        if memref_type.get_num_dims() != len(self.indices):
            raise Exception("Expected an index for each dimension")

    @classmethod
    def get(
        cls,
        value: Operation | SSAValue,
        ref: Operation | SSAValue,
        indices: Sequence[Operation | SSAValue],
    ) -> Self:
        return cls(operands=[value, ref, indices])


@irdl_op_definition
class Alloc(IRDLOperation):
    name = "memref.alloc"

    dynamic_sizes: VarOperand = var_operand_def(IndexType)
    symbol_operands: VarOperand = var_operand_def(IndexType)

    memref: OpResult = result_def(MemRefType[Attribute])

    # TODO how to constraint the IntegerAttr type?
    alignment: AnyIntegerAttr | None = opt_prop_def(AnyIntegerAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        dynamic_sizes: Sequence[SSAValue],
        symbol_operands: Sequence[SSAValue],
        result_type: Attribute,
        alignment: Attribute | None = None,
    ):
        super().__init__(
            operands=(dynamic_sizes, symbol_operands),
            result_types=(result_type,),
            properties={"alignment": alignment},
        )

    @classmethod
    def get(
        cls,
        return_type: Attribute,
        alignment: int | AnyIntegerAttr | None = None,
        shape: Iterable[int | IntAttr] | None = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
        layout: Attribute = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> Self:
        if shape is None:
            shape = [1]

        if dynamic_sizes is None:
            dynamic_sizes = []

        if isinstance(alignment, int):
            alignment = IntegerAttr.from_int_and_width(alignment, 64)

        return cls(
            tuple(SSAValue.get(ds) for ds in dynamic_sizes),
            (),
            MemRefType(return_type, shape, layout, memory_space),
            alignment,
        )

    def verify_(self) -> None:
        memref_type = self.memref.type
        if not isinstance(memref_type, MemRefType):
            raise VerifyException("expected result to be a memref")
        memref_type = cast(MemRefType[Attribute], memref_type)

        dyn_dims = [x for x in memref_type.shape.data if x.data == -1]
        if len(dyn_dims) != len(self.dynamic_sizes):
            raise VerifyException(
                "op dimension operand count does not equal memref dynamic dimension count."
            )

    def print(self, printer: Printer):
        printer.print_string("(")
        printer.print_list(self.dynamic_sizes, printer.print_ssa_value)
        printer.print_string(")")
        if self.symbol_operands:
            printer.print_string("[")
            printer.print_list(self.symbol_operands, printer.print_ssa_value)
            printer.print_string("]")

        printer.print_op_attributes(
            self.properties | self.attributes,
            print_keyword=False,
            reserved_attr_names="operandSegmentSizes",
        )

        printer.print_string(" : ")
        printer.print_attribute(self.memref.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        #  %alloc = memref.alloc(%a)[%s] {alignment = 64 : i64} : memref<3x2xf32>

        unresolved_dynamic_sizes = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        unresolved_symbol_operands = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_unresolved_operand
        )
        if unresolved_symbol_operands is None:
            unresolved_symbol_operands = []

        attrs = parser.parse_optional_attr_dict()

        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()

        index = IndexType()
        dynamic_sizes = tuple(
            parser.resolve_operand(uop, index) for uop in unresolved_dynamic_sizes
        )
        symbol_operands = tuple(
            parser.resolve_operand(uop, index) for uop in unresolved_symbol_operands
        )

        if "alignment" in attrs:
            alignment = attrs["alignment"]
            del attrs["alignment"]
        else:
            alignment = None

        op = cls(
            dynamic_sizes,
            symbol_operands,
            res_type,
            alignment,
        )

        op.attributes |= attrs

        return op


@irdl_op_definition
class AllocaScopeOp(IRDLOperation):
    name = "memref.alloca_scope"

    res = var_result_def()

    scope = region_def()


@irdl_op_definition
class AllocaScopeReturnOp(IRDLOperation):
    name = "memref.alloca_scope.return"

    ops = var_operand_def()

    traits = frozenset([IsTerminator(), HasParent(AllocaScopeOp)])

    def verify_(self) -> None:
        parent = cast(AllocaScopeOp, self.parent_op())
        if any(op.type != res.type for op, res in zip(self.ops, parent.results)):
            raise VerifyException(
                "Expected operand types to match parent's return types."
            )


@irdl_op_definition
class Alloca(IRDLOperation):
    name = "memref.alloca"

    dynamic_sizes: VarOperand = var_operand_def(IndexType)
    symbol_operands: VarOperand = var_operand_def(IndexType)

    memref: OpResult = result_def(MemRefType[Attribute])

    # TODO how to constraint the IntegerAttr type?
    alignment: AnyIntegerAttr | None = opt_prop_def(AnyIntegerAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    @staticmethod
    def get(
        return_type: Attribute,
        alignment: int | AnyIntegerAttr | None = None,
        shape: Iterable[int | IntAttr] | None = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
        layout: Attribute = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> Alloca:
        if shape is None:
            shape = [1]

        if dynamic_sizes is None:
            dynamic_sizes = []

        if isinstance(alignment, int):
            alignment = IntegerAttr.from_int_and_width(alignment, 64)

        return Alloca.build(
            operands=[dynamic_sizes, []],
            result_types=[MemRefType(return_type, shape, layout, memory_space)],
            properties={
                "alignment": alignment,
            },
        )

    def verify_(self) -> None:
        memref_type = self.memref.type
        if not isinstance(memref_type, MemRefType):
            raise VerifyException("expected result to be a memref")
        memref_type = cast(MemRefType[Attribute], memref_type)

        dyn_dims = [x for x in memref_type.shape.data if x.data == -1]
        if len(dyn_dims) != len(self.dynamic_sizes):
            raise VerifyException(
                "op dimension operand count does not equal memref dynamic dimension count."
            )


@irdl_op_definition
class Dealloc(IRDLOperation):
    name = "memref.dealloc"
    memref: Operand = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    @staticmethod
    def get(operand: Operation | SSAValue) -> Dealloc:
        return Dealloc.build(operands=[operand])

    assembly_format = "$memref attr-dict `:` type($memref)"


@irdl_op_definition
class GetGlobal(IRDLOperation):
    name = "memref.get_global"
    memref: OpResult = result_def(MemRefType[Attribute])
    name_: SymbolRefAttr = prop_def(SymbolRefAttr, prop_name="name")

    @staticmethod
    def get(name: str, return_type: Attribute) -> GetGlobal:
        return GetGlobal.build(
            result_types=[return_type], properties={"name": SymbolRefAttr(name)}
        )

    assembly_format = "$name `:` type($memref) attr-dict"

    # TODO how to verify the types, as the global might be defined in another
    # compilation unit


@irdl_op_definition
class Global(IRDLOperation):
    name = "memref.global"

    sym_name: StringAttr = prop_def(StringAttr)
    sym_visibility: StringAttr = prop_def(StringAttr)
    type: Attribute = prop_def(Attribute)
    initial_value: Attribute = prop_def(Attribute)
    constant = opt_prop_def(UnitAttr)
    alignment = opt_prop_def(IntegerAttr[Annotated[IntegerType, IntegerType(64)]])

    traits = frozenset([SymbolOpInterface()])

    def verify_(self) -> None:
        if not isinstance(self.type, MemRefType):
            raise Exception("Global expects a MemRefType")

        if not isinstance(self.initial_value, UnitAttr | DenseIntOrFPElementsAttr):
            raise Exception(
                "Global initial value is expected to be a "
                "dense type or an unit attribute"
            )
        if self.alignment is not None:
            assert isinstance(self.alignment, IntegerAttr)
            alignment_value = self.alignment.value.data
            # Alignment has to be a power of two
            if not (is_power_of_two(alignment_value)):
                raise VerifyException(
                    f"Alignment attribute {alignment_value} is not a power of 2"
                )

    @staticmethod
    def get(
        sym_name: StringAttr,
        sym_type: Attribute,
        initial_value: Attribute,
        sym_visibility: StringAttr = StringAttr("private"),
        constant: UnitAttr | None = None,
        alignment: int | IntegerAttr[IntegerType] | None = None,
    ) -> Global:
        if isinstance(alignment, int):
            alignment = IntegerAttr.from_int_and_width(alignment, 64)

        return Global.build(
            properties={
                "sym_name": sym_name,
                "type": sym_type,
                "initial_value": initial_value,
                "sym_visibility": sym_visibility,
                "constant": constant,
                "alignment": alignment,
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


ReassociationAttr = ArrayAttr[
    ArrayAttr[IntegerAttr[Annotated[IntegerType, IntegerType(64)]]]
]


class AlterShapeOp(IRDLOperation):
    src: Operand = operand_def(MemRefType)
    result: OpResult = result_def(MemRefType)
    reassociation = prop_def(ReassociationAttr)
    assembly_format = (
        "$src $reassociation attr-dict `:` type($src) `into` type($result)"
    )


@irdl_op_definition
class CollapseShapeOp(AlterShapeOp):
    """
    https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcollapse_shape-memrefcollapseshapeop
    """

    name = "memref.collapse_shape"


@irdl_op_definition
class ExpandShapeOp(AlterShapeOp):
    """
    https://mlir.llvm.org/docs/Dialects/MemRef/#memrefexpand_shape-memrefexpandshapeop
    """

    name = "memref.expand_shape"


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


class MemrefHasCanonicalizationPatternsTrait(HasCanonicalisationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.memref import (
            MemrefSubviewOfSubviewFolding,
        )

        return (MemrefSubviewOfSubviewFolding(),)


@irdl_op_definition
class Subview(IRDLOperation):
    name = "memref.subview"

    source: Operand = operand_def(MemRefType)
    offsets: VarOperand = var_operand_def(IndexType)
    sizes: VarOperand = var_operand_def(IndexType)
    strides: VarOperand = var_operand_def(IndexType)
    static_offsets: DenseArrayBase = prop_def(DenseArrayBase)
    static_sizes: DenseArrayBase = prop_def(DenseArrayBase)
    static_strides: DenseArrayBase = prop_def(DenseArrayBase)
    result: OpResult = result_def(MemRefType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset((MemrefHasCanonicalizationPatternsTrait(),))

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

        source_shape = source_type.get_shape()
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

        return_type = MemRefType(
            source_type.element_type,
            result_sizes,
            layout,
            source_type.memory_space,
        )

        return Subview.build(
            operands=[source, [], [], []],
            result_types=[return_type],
            properties={
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
class MemorySpaceCast(IRDLOperation):
    name = "memref.memory_space_cast"

    source = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])
    dest = result_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: MemRefType[Attribute] | UnrankedMemrefType[Attribute],
    ):
        super().__init__(operands=[source], result_types=[dest])

    @staticmethod
    def from_type_and_target_space(
        source: SSAValue | Operation,
        type: MemRefType[Attribute],
        dest_memory_space: Attribute,
    ) -> MemorySpaceCast:
        dest = MemRefType(
            type.get_element_type(),
            shape=type.get_shape(),
            layout=type.layout,
            memory_space=dest_memory_space,
        )
        return MemorySpaceCast(source, dest)

    def verify_(self) -> None:
        source = cast(MemRefType[Attribute], self.source.type)
        dest = cast(MemRefType[Attribute], self.dest.type)
        if source.get_shape() != dest.get_shape():
            raise VerifyException(
                "Expected source and destination to have the same shape."
            )
        if source.get_element_type() != dest.get_element_type():
            raise VerifyException(
                "Expected source and destination to have the same element type."
            )


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
                f"Expected {len(self.src.type.shape)} source indices (because of shape of src memref)"
            )

        if len(self.dest.type.shape) != len(self.dest_indices):
            raise VerifyException(
                f"Expected {len(self.dest.type.shape)} dest indices (because of shape of dest memref)"
            )

        if len(self.tag.type.shape) != len(self.tag_indices):
            raise VerifyException(
                f"Expected {len(self.tag.type.shape)} tag indices (because of shape of tag memref)"
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
        super().__init__(operands=[source, destination])

    def verify_(self) -> None:
        source = cast(MemRefType[Attribute], self.source.type)
        destination = cast(MemRefType[Attribute], self.destination.type)
        if source.get_shape() != destination.get_shape():
            raise VerifyException(
                "Expected source and destination to have the same shape."
            )
        if source.get_element_type() != destination.get_element_type():
            raise VerifyException(
                "Expected source and destination to have the same element type."
            )


MemRef = Dialect(
    "memref",
    [
        Load,
        Store,
        Alloc,
        Alloca,
        AllocaScopeOp,
        AllocaScopeReturnOp,
        CopyOp,
        CollapseShapeOp,
        ExpandShapeOp,
        Dealloc,
        GetGlobal,
        Global,
        Dim,
        ExtractAlignedPointerAsIndexOp,
        Subview,
        Cast,
        MemorySpaceCast,
        DmaStartOp,
        DmaWaitOp,
    ],
    [],
)
