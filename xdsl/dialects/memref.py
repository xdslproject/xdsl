from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from typing import ClassVar, cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    I64,
    AnyFloatConstr,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefLayoutAttr,
    MemRefType,
    NoneAttr,
    SignlessIntegerConstraint,
    StridedLayoutAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    UnrankedMemRefType,
    i32,
    i64,
)
from xdsl.dialects.utils import (
    split_dynamic_index_list,
)
from xdsl.dialects.utils.dynamic_index_list import (
    DynamicIndexList,
    verify_dynamic_index_list,
)
from xdsl.dialects.utils.reshape_ops_utils import (
    ContiguousArrayOfIntArray,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    SameVariadicResultSize,
    VarConstraint,
    base,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    HasParent,
    IsTerminator,
    MemoryAllocEffect,
    MemoryFreeEffect,
    MemoryReadEffect,
    MemoryWriteEffect,
    NoMemoryEffect,
    SymbolOpInterface,
)
from xdsl.utils.bitwise_casts import is_power_of_two
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "memref.load"

    T: ClassVar = VarConstraint("T", AnyAttr())

    nontemporal = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType())
    res = result_def(T)

    traits = traits_def(MemoryReadEffect())

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = "$memref `[` $indices `]` attr-dict `:` type($memref)"

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters,
    # which is subject to change

    def verify_(self):
        memref_type = self.memref.type
        if not isinstance(memref_type, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_type = cast(MemRefType, memref_type)

        if memref_type.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @classmethod
    def get(
        cls, ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]
    ) -> Self:
        ssa_value = SSAValue.get(ref, type=MemRefType)
        return cls(operands=[ref, indices], result_types=[ssa_value.type.element_type])


@irdl_op_definition
class StoreOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "memref.store"

    nontemporal = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    value = operand_def(T)
    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType())

    traits = traits_def(MemoryWriteEffect())

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref)"

    def verify_(self):
        if not isinstance(memref_type := self.memref.type, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_type = cast(MemRefType, memref_type)

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


class AllocOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.memref import ElideUnusedAlloc

        return (ElideUnusedAlloc(),)


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "memref.alloc"

    dynamic_sizes = var_operand_def(IndexType)
    symbol_operands = var_operand_def(IndexType)

    memref = result_def(MemRefType)

    # TODO how to constraint the IntegerAttr type?
    alignment = opt_prop_def(IntegerAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(AllocOpHasCanonicalizationPatterns(), MemoryAllocEffect())

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
        alignment: int | IntegerAttr | None = None,
        shape: Iterable[int | IntAttr] | None = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
        layout: MemRefLayoutAttr | NoneAttr = NoneAttr(),
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

        dyn_dims = [x for x in memref_type.shape.data if x.data == DYNAMIC_INDEX]
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

    traits = traits_def(IsTerminator(), HasParent(AllocaScopeOp))

    def verify_(self) -> None:
        parent = cast(AllocaScopeOp, self.parent_op())
        if self.ops.types != parent.result_types:
            raise VerifyException(
                "Expected operand types to match parent's return types."
            )


@irdl_op_definition
class AllocaOp(IRDLOperation):
    name = "memref.alloca"

    dynamic_sizes = var_operand_def(IndexType)
    symbol_operands = var_operand_def(IndexType)

    memref = result_def(MemRefType)

    # TODO how to constraint the IntegerAttr type?
    alignment = opt_prop_def(IntegerAttr)

    traits = traits_def(MemoryAllocEffect())

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    @staticmethod
    def get(
        return_type: Attribute,
        alignment: int | IntegerAttr | None = None,
        shape: Iterable[int | IntAttr] | None = None,
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
        layout: MemRefLayoutAttr | NoneAttr = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> AllocaOp:
        if shape is None:
            shape = [1]

        if dynamic_sizes is None:
            dynamic_sizes = []

        if isinstance(alignment, int):
            alignment = IntegerAttr.from_int_and_width(alignment, 64)

        return AllocaOp.build(
            operands=[dynamic_sizes, []],
            result_types=[MemRefType(return_type, shape, layout, memory_space)],
            properties={
                "alignment": alignment,
            },
        )

    def verify_(self) -> None:
        memref_type = self.memref.type

        dyn_dims = [x for x in memref_type.shape.data if x.data == DYNAMIC_INDEX]
        if len(dyn_dims) != len(self.dynamic_sizes):
            raise VerifyException(
                "op dimension operand count does not equal memref dynamic dimension count."
            )


@irdl_op_definition
class AtomicRMWOp(IRDLOperation):
    name = "memref.atomic_rmw"

    T: ClassVar = VarConstraint("T", AnyFloatConstr | SignlessIntegerConstraint)

    value = operand_def(T)
    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)

    kind = prop_def(IntegerAttr[I64])

    result = result_def(T)

    traits = traits_def(MemoryWriteEffect(), MemoryReadEffect())


@irdl_op_definition
class DeallocOp(IRDLOperation):
    name = "memref.dealloc"
    memref = operand_def(base(MemRefType) | base(UnrankedMemRefType))

    traits = traits_def(MemoryFreeEffect())

    @staticmethod
    def get(operand: Operation | SSAValue) -> DeallocOp:
        return DeallocOp.build(operands=[operand])

    assembly_format = "$memref attr-dict `:` type($memref)"


@irdl_op_definition
class GetGlobalOp(IRDLOperation):
    name = "memref.get_global"
    memref = result_def(MemRefType)
    name_ = prop_def(SymbolRefAttr, prop_name="name")

    traits = traits_def(NoMemoryEffect())

    assembly_format = "$name `:` type($memref) attr-dict"

    def __init__(self, name: str | SymbolRefAttr, return_type: Attribute):
        if isinstance(name, str):
            name = SymbolRefAttr(name)
        super().__init__(result_types=[return_type], properties={"name": name})

    # TODO how to verify the types, as the global might be defined in another
    # compilation unit


@irdl_op_definition
class GlobalOp(IRDLOperation):
    name = "memref.global"

    sym_name = prop_def(SymbolNameConstraint())
    sym_visibility = prop_def(StringAttr)
    type = prop_def(MemRefType)
    initial_value = prop_def(UnitAttr | DenseIntOrFPElementsAttr)
    constant = opt_prop_def(UnitAttr)
    alignment = opt_prop_def(IntegerAttr[I64])

    traits = traits_def(SymbolOpInterface(), MemoryAllocEffect())

    def verify_(self) -> None:
        if self.alignment is not None:
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
    ) -> GlobalOp:
        if isinstance(alignment, int):
            alignment = IntegerAttr.from_int_and_width(alignment, 64)

        return GlobalOp.build(
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
class DimOp(IRDLOperation):
    name = "memref.dim"

    source = operand_def(base(MemRefType) | base(UnrankedMemRefType))
    index = operand_def(IndexType)

    result = result_def(IndexType)

    traits = traits_def(NoMemoryEffect())

    @staticmethod
    def from_source_and_index(
        source: SSAValue | Operation, index: SSAValue | Operation
    ):
        return DimOp.build(operands=[source, index], result_types=[IndexType()])


@irdl_op_definition
class RankOp(IRDLOperation):
    name = "memref.rank"

    source = operand_def(MemRefType)

    rank = result_def(IndexType)

    traits = traits_def(NoMemoryEffect())

    @staticmethod
    def from_memref(memref: Operation | SSAValue):
        return RankOp.build(operands=[memref], result_types=[IndexType()])


class AlterShapeOperation(IRDLOperation, abc.ABC):
    result = result_def(MemRefType)
    reassociation = prop_def(ContiguousArrayOfIntArray())

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class CollapseShapeOp(AlterShapeOperation):
    """
    https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcollapse_shape-memrefcollapseshapeop
    """

    name = "memref.collapse_shape"

    src = operand_def(MemRefType)

    assembly_format = (
        "$src $reassociation attr-dict `:` type($src) `into` type($result)"
    )


@irdl_op_definition
class ExpandShapeOp(AlterShapeOperation):
    """
    https://mlir.llvm.org/docs/Dialects/MemRef/#memrefexpand_shape-memrefexpandshapeop
    """

    name = "memref.expand_shape"

    src = operand_def(MemRefType)
    output_shape = var_operand_def(IndexType)

    static_output_shape = prop_def(DenseArrayBase.constr(i64))

    assembly_format = (
        "$src $reassociation `output_shape`"
        "custom<DynamicIndexList>($output_shape, $static_output_shape) attr-dict `:`"
        "type($src) `into` type($result)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ExtractStridedMetaDataOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_strided_metadata-memrefextractstridedmetadataop
    """

    name = "memref.extract_strided_metadata"

    source = operand_def(MemRefType)

    base_buffer = result_def(MemRefType)
    offset = result_def(IndexType)
    sizes = var_result_def(IndexType)
    strides = var_result_def(IndexType)

    traits = traits_def(NoMemoryEffect())

    irdl_options = [SameVariadicResultSize()]

    def __init__(self, source: SSAValue | Operation):
        """
        Create an ExtractStridedMetaDataOp that extracts the metadata from the
        operation (source) that produces a memref.
        """
        source_type = SSAValue.get(source, type=MemRefType).type
        source_shape = source_type.get_shape()
        # Return a rank zero memref with the memref type
        base_buffer_type = MemRefType(
            source_type.element_type,
            [],
            NoneAttr(),
            source_type.memory_space,
        )
        offset_type = IndexType()
        # There are as many strides/sizes as there are shape dimensions
        strides_type = [IndexType()] * len(source_shape)
        sizes_type = [IndexType()] * len(source_shape)
        return_type = [base_buffer_type, offset_type, strides_type, sizes_type]
        super().__init__(operands=[source], result_types=return_type)


@irdl_op_definition
class ExtractAlignedPointerAsIndexOp(IRDLOperation):
    name = "memref.extract_aligned_pointer_as_index"

    source = operand_def(MemRefType)

    aligned_pointer = result_def(IndexType)

    traits = traits_def(NoMemoryEffect())

    @staticmethod
    def get(source: SSAValue | Operation):
        return ExtractAlignedPointerAsIndexOp.build(
            operands=[source], result_types=[IndexType()]
        )


class MemRefHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.memref import (
            MemRefSubviewOfSubviewFolding,
        )

        return (MemRefSubviewOfSubviewFolding(),)


@irdl_op_definition
class SubviewOp(IRDLOperation):
    DYNAMIC_INDEX: ClassVar[int] = -9223372036854775808
    """
    Constant value used to denote dynamic indices in offsets, sizes, and strides.
    Same constant as in MLIR.
    """

    name = "memref.subview"

    source = operand_def(MemRefType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    static_offsets = prop_def(DenseArrayBase.constr(i64))
    static_sizes = prop_def(DenseArrayBase.constr(i64))
    static_strides = prop_def(DenseArrayBase.constr(i64))
    result = result_def(MemRefType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = lazy_traits_def(
        lambda: (MemRefHasCanonicalizationPatternsTrait(), NoMemoryEffect())
    )

    assembly_format = (
        "$source ``"
        "custom<DynamicIndexList>($offsets, $static_offsets)"
        "custom<DynamicIndexList>($sizes, $static_sizes)"
        "custom<DynamicIndexList>($strides, $static_strides)"
        "attr-dict `:` type($source) `to` type($result)"
    )

    custom_directives = (DynamicIndexList,)

    def verify_(self) -> None:
        static_offsets = self.static_offsets.get_values()
        static_sizes = self.static_sizes.get_values()
        static_strides = self.static_strides.get_values()
        verify_dynamic_index_list(
            static_sizes, self.sizes, self.DYNAMIC_INDEX, " in the size arguments"
        )
        verify_dynamic_index_list(
            static_offsets, self.offsets, self.DYNAMIC_INDEX, " in the offset arguments"
        )
        verify_dynamic_index_list(
            static_strides, self.strides, self.DYNAMIC_INDEX, " in the stride arguments"
        )

    def __init__(
        self,
        source: SSAValue | Operation,
        offsets: Sequence[SSAValue],
        sizes: Sequence[SSAValue],
        strides: Sequence[SSAValue],
        static_offsets: Sequence[int] | DenseArrayBase,
        static_sizes: Sequence[int] | DenseArrayBase,
        static_strides: Sequence[int] | DenseArrayBase,
        result_type: Attribute,
    ):
        if not isinstance(static_offsets, DenseArrayBase):
            static_offsets = DenseArrayBase.from_list(i64, static_offsets)
        if not isinstance(static_sizes, DenseArrayBase):
            static_sizes = DenseArrayBase.from_list(i64, static_sizes)
        if not isinstance(static_strides, DenseArrayBase):
            static_strides = DenseArrayBase.from_list(i64, static_strides)
        super().__init__(
            operands=[source, offsets, sizes, strides],
            result_types=[result_type],
            properties={
                "static_offsets": static_offsets,
                "static_sizes": static_sizes,
                "static_strides": static_strides,
            },
        )

    @staticmethod
    def get(
        source: SSAValue,
        offsets: Sequence[SSAValue | int],
        sizes: Sequence[SSAValue | int],
        strides: Sequence[SSAValue | int],
        result_type: Attribute,
    ) -> SubviewOp:
        static_offsets, dyn_offsets = split_dynamic_index_list(
            offsets, SubviewOp.DYNAMIC_INDEX
        )
        static_sizes, dyn_sizes = split_dynamic_index_list(
            sizes, SubviewOp.DYNAMIC_INDEX
        )
        static_strides, dyn_strides = split_dynamic_index_list(
            strides, SubviewOp.DYNAMIC_INDEX
        )

        return SubviewOp(
            source,
            dyn_offsets,
            dyn_sizes,
            dyn_strides,
            static_offsets,
            static_sizes,
            static_strides,
            result_type,
        )

    @staticmethod
    def from_static_parameters(
        source: SSAValue | Operation,
        source_type: MemRefType,
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int],
        reduce_rank: bool = False,
    ) -> SubviewOp:
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

        return SubviewOp(
            source,
            (),
            (),
            (),
            DenseArrayBase.from_list(i64, offsets),
            DenseArrayBase.from_list(i64, sizes),
            DenseArrayBase.from_list(i64, strides),
            return_type,
        )


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "memref.cast"

    source = operand_def(base(MemRefType) | base(UnrankedMemRefType))
    dest = result_def(base(MemRefType) | base(UnrankedMemRefType))

    traits = traits_def(NoMemoryEffect())

    @staticmethod
    def get(
        source: SSAValue | Operation,
        type: MemRefType | UnrankedMemRefType,
    ):
        return CastOp.build(operands=[source], result_types=[type])


@irdl_op_definition
class MemorySpaceCastOp(IRDLOperation):
    name = "memref.memory_space_cast"

    source = operand_def(base(MemRefType) | base(UnrankedMemRefType))
    dest = result_def(base(MemRefType) | base(UnrankedMemRefType))

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: MemRefType | UnrankedMemRefType,
    ):
        super().__init__(operands=[source], result_types=[dest])

    @staticmethod
    def from_type_and_target_space(
        source: SSAValue | Operation,
        type: MemRefType,
        dest_memory_space: Attribute,
    ) -> MemorySpaceCastOp:
        dest = MemRefType(
            type.get_element_type(),
            shape=type.get_shape(),
            layout=type.layout,
            memory_space=dest_memory_space,
        )
        return MemorySpaceCastOp(source, dest)

    def verify_(self) -> None:
        source = cast(MemRefType, self.source.type)
        dest = cast(MemRefType, self.dest.type)
        if source.get_shape() != dest.get_shape():
            raise VerifyException(
                "Expected source and destination to have the same shape."
            )
        if source.get_element_type() != dest.get_element_type():
            raise VerifyException(
                "Expected source and destination to have the same element type."
            )


@irdl_op_definition
class ReinterpretCastOp(IRDLOperation):
    DYNAMIC_INDEX: ClassVar[int] = -9223372036854775808

    name = "memref.reinterpret_cast"

    source = operand_def(MemRefType)

    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)

    static_offsets = prop_def(DenseArrayBase.constr(i64))
    static_sizes = prop_def(DenseArrayBase.constr(i64))
    static_strides = prop_def(DenseArrayBase.constr(i64))

    result = result_def(MemRefType)

    traits = traits_def(NoMemoryEffect())

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    assembly_format = (
        "$source `to` `offset` `` `:`"
        "custom<DynamicIndexList>($offsets, $static_offsets)"
        "`` `,` `sizes` `` `:`"
        "custom<DynamicIndexList>($sizes, $static_sizes)"
        "`` `,` `strides` `` `:`"
        "custom<DynamicIndexList>($strides, $static_strides)"
        "attr-dict `:` type($source) `to` type($result)"
    )

    custom_directives = (DynamicIndexList,)

    def __init__(
        self,
        source: SSAValue | Operation,
        offsets: Sequence[SSAValue],
        sizes: Sequence[SSAValue],
        strides: Sequence[SSAValue],
        static_offsets: Sequence[int] | DenseArrayBase,
        static_sizes: Sequence[int] | DenseArrayBase,
        static_strides: Sequence[int] | DenseArrayBase,
        result_type: Attribute,
    ):
        if not isinstance(static_offsets, DenseArrayBase):
            static_offsets = DenseArrayBase.from_list(i64, static_offsets)
        if not isinstance(static_sizes, DenseArrayBase):
            static_sizes = DenseArrayBase.from_list(i64, static_sizes)
        if not isinstance(static_strides, DenseArrayBase):
            static_strides = DenseArrayBase.from_list(i64, static_strides)
        super().__init__(
            operands=[source, offsets, sizes, strides],
            result_types=[result_type],
            properties={
                "static_offsets": static_offsets,
                "static_sizes": static_sizes,
                "static_strides": static_strides,
            },
        )

    @staticmethod
    def from_dynamic(
        source: SSAValue,
        offsets: Sequence[SSAValue | int],
        sizes: Sequence[SSAValue | int],
        strides: Sequence[SSAValue | int],
        result_type: Attribute,
    ):
        """
        Construct a `ReinterpretCastOp` from dynamic offsets, sizes, and strides.
        """
        static_offsets, dyn_offsets = split_dynamic_index_list(
            offsets, ReinterpretCastOp.DYNAMIC_INDEX
        )
        static_sizes, dyn_sizes = split_dynamic_index_list(
            sizes, ReinterpretCastOp.DYNAMIC_INDEX
        )
        static_strides, dyn_strides = split_dynamic_index_list(
            strides, ReinterpretCastOp.DYNAMIC_INDEX
        )

        return ReinterpretCastOp(
            source,
            dyn_offsets,
            dyn_sizes,
            dyn_strides,
            static_offsets,
            static_sizes,
            static_strides,
            result_type,
        )

    def verify_(self):
        static_offsets = self.static_offsets.get_values()
        static_sizes = self.static_sizes.get_values()
        static_strides = self.static_strides.get_values()

        verify_dynamic_index_list(
            static_sizes, self.sizes, self.DYNAMIC_INDEX, " in the size arguments"
        )
        verify_dynamic_index_list(
            static_offsets, self.offsets, self.DYNAMIC_INDEX, " in the offset arguments"
        )
        verify_dynamic_index_list(
            static_strides, self.strides, self.DYNAMIC_INDEX, " in the stride arguments"
        )

        assert isa(self.source.type, MemRefType)
        assert isa(self.result.type, MemRefType)

        if len(self.result.type.shape) != len(self.static_sizes):
            raise VerifyException(
                f"Expected {len(self.source.type.shape)} size values but got {len(self.static_sizes)}"
            )

        # validate sizes
        for dim, (actual, expected) in enumerate(
            zip(
                self.result.type.get_shape(),
                self.static_sizes.get_values(),
                strict=True,
            )
        ):
            if expected == ReinterpretCastOp.DYNAMIC_INDEX and actual != DYNAMIC_INDEX:
                raise VerifyException(
                    f"Expected result type with dynamic size instead of {actual} in dim = {dim}"
                )
            elif expected != ReinterpretCastOp.DYNAMIC_INDEX and expected != actual:
                raise VerifyException(
                    f"Expected result type with size = {expected} instead of {actual} in dim = {dim}"
                )


@irdl_op_definition
class DmaStartOp(IRDLOperation):
    name = "memref.dma_start"

    src = operand_def(MemRefType)
    src_indices = var_operand_def(IndexType)

    dest = operand_def(MemRefType)
    dest_indices = var_operand_def(IndexType)

    num_elements = operand_def(IndexType)

    tag = operand_def(MemRefType[IntegerType])
    tag_indices = var_operand_def(IndexType)

    traits = traits_def(MemoryWriteEffect(), MemoryReadEffect())

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
        assert isa(self.src.type, MemRefType)
        assert isa(self.dest.type, MemRefType)
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

    tag = operand_def(MemRefType)
    tag_indices = var_operand_def(IndexType)

    num_elements = operand_def(IndexType)

    traits = traits_def(MemoryWriteEffect(), MemoryReadEffect())

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
        assert isa(self.tag.type, MemRefType)

        if len(self.tag.type.shape) != len(self.tag_indices):
            raise VerifyException(
                f"Expected {len(self.tag.type.shape)} tag indices because of shape of tag memref"
            )

        if self.tag.type.element_type != i32:
            raise VerifyException("Expected tag to be a memref of i32")


@irdl_op_definition
class CopyOp(IRDLOperation):
    name = "memref.copy"
    source = operand_def(MemRefType)
    destination = operand_def(MemRefType)

    traits = traits_def(MemoryWriteEffect(), MemoryReadEffect())

    def __init__(self, source: SSAValue | Operation, destination: SSAValue | Operation):
        super().__init__(operands=[source, destination])

    def verify_(self) -> None:
        source = cast(MemRefType, self.source.type)
        destination = cast(MemRefType, self.destination.type)
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
        LoadOp,
        StoreOp,
        AllocOp,
        AllocaOp,
        AllocaScopeOp,
        AllocaScopeReturnOp,
        AtomicRMWOp,
        CopyOp,
        CollapseShapeOp,
        ExpandShapeOp,
        DeallocOp,
        GetGlobalOp,
        GlobalOp,
        DimOp,
        ExtractStridedMetaDataOp,
        ExtractAlignedPointerAsIndexOp,
        SubviewOp,
        CastOp,
        MemorySpaceCastOp,
        ReinterpretCastOp,
        DmaStartOp,
        DmaWaitOp,
        RankOp,
    ],
    [],
)
