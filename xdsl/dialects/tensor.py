from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnySignlessIntegerOrIndexType,
    ArrayAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    Region,
    ShapedType,
    TensorType,
    UnitAttr,
    UnrankedTensorType,
    i64,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
)
from xdsl.dialects.utils.dynamic_index_list import (
    DynamicIndexList,
    parse_dynamic_index_list_without_types,
    print_dynamic_index_list,
)
from xdsl.dialects.utils.reshape_ops_utils import (
    ContiguousArrayOfIntArray,
    verify_reshape_like_types,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import AlwaysSpeculatable, IsTerminator, NoMemoryEffect, Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    Tensor cast operation.

    Convert a tensor from one type to an equivalent type without changing any data elements.
    The source and destination types must both be tensor types with the same element type.
    If both are ranked, then the rank should be the same and static dimensions should match.
    The operation is invalid if converting to a mismatching constant dimension.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcast-tensorcastop
    """

    name = "tensor.cast"

    source = operand_def(
        base(TensorType[Attribute]) | base(UnrankedTensorType[Attribute])
    )
    dest = result_def(base(TensorType[Attribute]) | base(UnrankedTensorType[Attribute]))

    assembly_format = "$source attr-dict `:` type($source) `to` type($dest)"

    traits = traits_def(NoMemoryEffect())

    def __init__(self, source: SSAValue | Operation, dest: TensorType[Attribute]):
        super().__init__(operands=(source,), result_types=(dest,))

    def verify_(self):
        source_type = self.source.type
        dest_type = self.dest.type

        if isinstance(source_type, TensorType) and isinstance(dest_type, TensorType):
            # rank should be the same + constant shapes equal
            if len(source_type.get_shape()) != (len(dest_type.get_shape())):
                raise VerifyException("source and destination rank should be the same")
            for a, b in zip(source_type.get_shape(), dest_type.get_shape()):
                if a >= 0 and b >= 0 and a != b:
                    raise VerifyException(
                        "source and destination constant dimensions should match"
                    )


@irdl_op_definition
class DimOp(IRDLOperation):
    """
    Dimension index operation.

    The tensor.dim operation takes a tensor and a dimension operand of type index.
    It returns the size of the requested dimension of the given tensor.
    If the dimension index is out of bounds, the behavior is undefined.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensordim-tensordimop
    """

    name = "tensor.dim"

    source = operand_def(
        base(TensorType[Attribute]) | base(UnrankedTensorType[Attribute])
    )
    index = operand_def(IndexType)
    result = result_def(IndexType)

    traits = traits_def(Pure())

    assembly_format = "attr-dict $source `,` $index `:` type($source)"

    def __init__(
        self,
        source: SSAValue | Operation,
        index: SSAValue | Operation,
        attributes: Mapping[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=(source, index), result_types=(IndexType(),), attributes=attributes
        )

    def verify_(self):
        if isinstance((source_type := self.source.type), TensorType):
            if not len(source_type.get_shape()):
                raise VerifyException("cannot get dim of 0-rank tensor")


@irdl_op_definition
class EmptyOp(IRDLOperation):
    """
    Empty tensor operation.

    Defines a tensor of a particular shape which could be dynamic or static.
    The contents of the tensor are unspecified and the only purpose of the op
    result is to materialize the specified shape in IR and make it available
    to other transformations.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-tensoremptyop
    """

    name = "tensor.empty"

    dynamic_sizes = var_operand_def(IndexType)

    tensor = result_def(TensorType[Attribute])

    traits = traits_def(NoMemoryEffect())

    def __init__(self, dynamic_sizes: Sequence[SSAValue], tensor_type: Attribute):
        super().__init__(
            operands=(dynamic_sizes,),
            result_types=(tensor_type,),
        )

    def print(self, printer: Printer):
        if self.dynamic_sizes:
            printer.print_string("(")
            printer.print_list(self.dynamic_sizes, printer.print_ssa_value)
            printer.print_string(")")
        else:
            printer.print_string("(")
            printer.print_string(")")

        printer.print_string(" : ")
        printer.print_attribute(self.tensor.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        pos = parser.pos
        parser.parse_punctuation("(")
        if parser.parse_optional_punctuation(")"):
            dynamic_sizes = ()
        else:
            unresolved_dynamic_sizes = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            unresolved_types = (IndexType(),) * len(unresolved_dynamic_sizes)
            parser.parse_punctuation(")")
            dynamic_sizes = parser.resolve_operands(
                unresolved_dynamic_sizes, unresolved_types, pos
            )
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()

        empty = cls(dynamic_sizes, result_type)

        return empty


@irdl_op_definition
class CollapseShapeOp(IRDLOperation):
    """
    Operation to produce a tensor with a smaller rank.

    The collapse_shape operation produces a new tensor of lower (or equal)
    rank whose dimension sizes are a reassociation of the original src dimensions.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcollapse_shape-tensorcollapseshapeop
    """

    name = "tensor.collapse_shape"

    src = operand_def(TensorType[Attribute])
    result = result_def(TensorType[Attribute])
    reassociation = prop_def(ContiguousArrayOfIntArray())
    assembly_format = (
        "$src $reassociation attr-dict `:` type($src) `into` type($result)"
    )

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    """
    Tensor reshape operation.

    The reshape operation converts a tensor from one type to an equivalent
    type with a provided shape. The source and destination types are compatible
    if both have the same element type, same number of elements.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorreshape-tensorreshapeop
    """

    name = "tensor.reshape"

    source = operand_def(TensorType[Attribute])
    shape = operand_def(TensorType[AnySignlessIntegerOrIndexType])
    result = result_def(TensorType[Attribute])
    assembly_format = "attr-dict $source `(` $shape `)` `:` `(` type($source) `,` type($shape) `)` `->` type($result)"

    traits = traits_def(NoMemoryEffect())

    def __init__(self, source: SSAValue, shape: SSAValue, result_type: Attribute):
        super().__init__(
            operands=(
                source,
                shape,
            ),
            result_types=(result_type,),
        )

    def verify_(self) -> None:
        if not isinstance(
            source_type := self.source.type, TensorType
        ) or not isinstance(shape_type := self.shape.type, TensorType):
            raise ValueError(
                "tensor elementwise operation operands and result must be of type TensorType"
            )

        source_type = cast(TensorType[Attribute], source_type)
        shape_type = cast(TensorType[Attribute], shape_type)
        res_type = self.result.type

        if source_type.element_type != res_type.element_type:
            raise VerifyException(
                "element types of source and result tensor types should be the same"
            )

        source_type = source_type.get_shape()
        shape_type = shape_type.get_shape()
        res_type = res_type.get_shape()

        if len(shape_type) != 1:
            raise VerifyException("shape tensor must have a rank one")

        # concerns the case of static reshaping
        if math.prod(source_type) != math.prod(res_type):
            raise VerifyException(
                "source and result tensor should have the same number of elements"
            )

        shape_size = shape_type[0]
        if shape_size != len(res_type):
            raise VerifyException(
                "length of shape operand differs from the result's tensor rank"
            )


@irdl_op_definition
class ExpandShapeOp(IRDLOperation):
    """
    Operation to produce a tensor with a higher rank.

    The tensor.expand_shape op produces a tensor of higher (or equal)
    rank than the operand src whose dimension sizes are a reassociation of src.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorexpand_shape-tensorexpandshapeop
    """

    # Constant value used to denote dynamic indices in offsets, sizes, and strides.
    # Same constant as in MLIR.
    DYNAMIC_INDEX: ClassVar[int] = -9223372036854775808

    name = "tensor.expand_shape"

    src = operand_def(TensorType)
    dynamic_output_shape = var_operand_def(IndexType)

    reassociation = prop_def(ContiguousArrayOfIntArray())

    static_output_shape = prop_def(DenseArrayBase.constr(i64))

    result = result_def(TensorType[Attribute])

    def __init__(
        self,
        src: SSAValue | Operation,
        dynamic_output_shape: Sequence[SSAValue],
        reassociation: ArrayAttr[ArrayAttr[IntegerAttr]],
        static_output_shape: Sequence[int] | DenseArrayBase,
        result_type: TensorType[Attribute],
        attributes: dict[str, Attribute] | None = None,
    ):
        if not isinstance(static_output_shape, DenseArrayBase):
            static_output_shape = DenseArrayBase.from_list(i64, static_output_shape)

        super().__init__(
            operands=[src, dynamic_output_shape],
            result_types=[result_type],
            properties={
                "reassociation": reassociation,
                "static_output_shape": static_output_shape,
            },
            attributes=attributes,
        )

    def verify_(self):
        assert isinstance(self.src.type, ShapedType)
        assert isinstance(self.result.type, ShapedType)

        # make sure the static output shape matches the result type
        if len(self.static_output_shape) != len(self.result.type.get_shape()):
            raise VerifyException(
                "expected number of static shape dims to be equal to the output rank "
                f"({len(self.result.type.get_shape())}) but found {len(self.static_output_shape)} inputs instead"
            )

        verify_reshape_like_types(
            collapsed_type=self.src.type,
            expanded_type=self.result.type,
            reassociation=self.reassociation,
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        src_operand = parser.parse_unresolved_operand()

        reassociation = parser.parse_attribute()
        parser.parse_characters("output_shape")
        index = IndexType()

        # Parse shape: mixture of ints and SSA values
        dyn_shape, static_shape = parse_dynamic_index_list_without_types(
            parser, dynamic_index=cls.DYNAMIC_INDEX
        )

        dyn_shape = parser.resolve_operands(
            dyn_shape, (index,) * len(dyn_shape), parser.pos
        )

        attributes = parser.parse_optional_attr_dict()

        parser.parse_punctuation(":")
        src_type = parser.parse_type()
        parser.parse_characters("into")
        result_type = parser.parse_type()
        src = parser.resolve_operand(src_operand, src_type)

        shape_attr = DenseArrayBase.from_list(i64, static_shape)

        reassociation = cast(ArrayAttr[ArrayAttr[IntegerAttr]], reassociation)
        result_type = cast(TensorType[Attribute], result_type)

        return cls(src, dyn_shape, reassociation, shape_attr, result_type, attributes)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.src)
        printer.print_string(" ")
        printer.print_attribute(self.reassociation)
        printer.print_string(" output_shape ")
        print_dynamic_index_list(
            printer,
            self.DYNAMIC_INDEX,
            self.dynamic_output_shape,
            self.static_output_shape.get_values(),
        )

        printer.print_op_attributes(attributes=self.attributes)

        printer.print_string(" : ")
        printer.print_attribute(self.src.type)
        printer.print_string(" into ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class ExtractSliceOp(IRDLOperation):
    """
    Extract slice operation.

    Extracts a tensor from another tensor as specified by the operation’s
    offsets, sizes and strides arguments.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract_slice-tensorextractsliceop
    """

    name = "tensor.extract_slice"

    source = operand_def(TensorType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    static_offsets = prop_def(DenseArrayBase.constr(i64))
    static_sizes = prop_def(DenseArrayBase.constr(i64))
    static_strides = prop_def(DenseArrayBase.constr(i64))
    result = result_def(TensorType)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = traits_def(NoMemoryEffect())

    @staticmethod
    def from_static_parameters(
        source: SSAValue | Operation,
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int] | None = None,
        reduce_rank: bool = False,
    ) -> ExtractSliceOp:
        if strides is None:
            strides = [1] * len(offsets)
        source_v = SSAValue.get(source, type=TensorType)
        source_t = source_v.type

        if reduce_rank:
            result_sizes = list(s for s in sizes if s != 1)
        else:
            result_sizes = list(sizes)

        return_type = TensorType(source_t.get_element_type(), result_sizes)

        return ExtractSliceOp.build(
            operands=[source, [], [], []],
            result_types=[return_type],
            properties={
                "static_offsets": DenseArrayBase.from_list(i64, offsets),
                "static_sizes": DenseArrayBase.from_list(i64, result_sizes),
                "static_strides": DenseArrayBase.from_list(i64, strides),
            },
        )


@irdl_op_definition
class InsertSliceOp(IRDLOperation):
    """
    Insert_slice operation.

    The insert_slice operation insert a tensor, source, into another tensor, dest,
    as specified by the operation’s offsets, sizes and strides arguments. It
    returns a copy of dest with the proper slice updated with the value of source.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert_slice-tensorinsertsliceop
    """

    name = "tensor.insert_slice"

    source = operand_def(TensorType)
    dest = operand_def(TensorType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    static_offsets = prop_def(DenseArrayBase.constr(i64))
    static_sizes = prop_def(DenseArrayBase.constr(i64))
    static_strides = prop_def(DenseArrayBase.constr(i64))
    result = result_def(TensorType)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = traits_def(NoMemoryEffect())

    @staticmethod
    def get(
        source: Operand,
        dest: Operand,
        static_sizes: Sequence[int],
        static_offsets: Sequence[int] | None = None,
        static_strides: Sequence[int] | None = None,
        offsets: Sequence[Operand] | None = None,
        sizes: Sequence[Operand] | None = None,
        strides: Sequence[Operand] | None = None,
        result_type: Attribute | None = None,
    ) -> InsertSliceOp:
        dims = len(static_sizes)
        offsets = [] if offsets is None else offsets
        sizes = [] if sizes is None else sizes
        strides = [] if strides is None else strides
        if not static_offsets:
            static_offsets = [DYNAMIC_INDEX] * len(offsets) + (
                [0] * (dims - len(offsets))
            )
        if not static_strides:
            static_strides = [DYNAMIC_INDEX] * len(strides) + (
                [1] * (dims - len(strides))
            )
        return InsertSliceOp.build(
            operands=[
                source,
                dest,
                offsets,
                sizes,
                strides,
            ],
            properties={
                "static_offsets": DenseArrayBase.from_list(
                    i64,
                    static_offsets,
                ),
                "static_sizes": DenseArrayBase.from_list(
                    i64,
                    static_sizes,
                ),
                "static_strides": DenseArrayBase.from_list(
                    i64,
                    static_strides,
                ),
            },
            result_types=[result_type if result_type else dest.type],
        )

    @staticmethod
    def from_static_parameters(
        source: SSAValue | Operation,
        dest: SSAValue | Operation,
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int] | None = None,
    ) -> InsertSliceOp:
        source = SSAValue.get(source)
        dest = SSAValue.get(dest)

        if strides is None:
            strides = [1] * len(sizes)

        return InsertSliceOp.build(
            operands=[source, dest, [], [], []],
            result_types=[dest.type],
            properties={
                "static_offsets": DenseArrayBase.from_list(i64, offsets),
                "static_sizes": DenseArrayBase.from_list(i64, sizes),
                "static_strides": DenseArrayBase.from_list(i64, strides),
            },
        )


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """
    Element extraction operation.

    The tensor.extract op reads a ranked tensor and returns one element as specified
    by the given indices. The result of the op is a value with the same type as the
    elements of the tensor. The arity of indices must match the rank of the accessed
    value. All indices should all be of index type.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract-tensorextractop
    """

    name = "tensor.extract"

    tensor = operand_def(TensorType)
    indices = var_operand_def(IndexType)
    result = result_def(Attribute)
    # assembly_format = "$tensor `[` $indices `]` attr-dict `:` type($tensor)"

    def __init__(
        self,
        tensor: SSAValue,
        indices: Sequence[SSAValue] | SSAValue,
        result_type: Attribute,
    ):
        if isinstance(indices, SSAValue):
            indices = [indices]
        return super().__init__(operands=[tensor, indices], result_types=[result_type])

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.tensor)
        printer.print_string("[")
        printer.print_list(self.indices, printer.print_ssa_value)
        printer.print_string("]")
        printer.print_string(" : ")
        printer.print_attribute(self.tensor.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        tensor = parser.parse_operand()
        indices = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.SQUARE, parse=parser.parse_operand
        )
        parser.parse_punctuation(":")
        source_tensor_type = parser.parse_type()
        tensor_type = cast(TensorType[Attribute], source_tensor_type)
        return cls(tensor, indices, tensor_type.get_element_type())


@irdl_op_definition
class InsertOp(IRDLOperation):
    """
    Element insertion operation.

    The tensor.insert op inserts a scalar into a ranked tensor, dest, as
    specified by the operation’s indices. It returns a copy of dest with the
    indexed position updated to the value of scalar.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert-tensorinsertop
    """

    name = "tensor.insert"

    scalar = operand_def(Attribute)
    dest = operand_def(TensorType)
    indices = var_operand_def(IndexType)
    result = result_def(TensorType)
    # assembly_format = "$scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)"

    def __init__(
        self,
        scalar: SSAValue,
        dest: SSAValue,
        indices: Sequence[SSAValue] | SSAValue,
    ):
        if isinstance(indices, SSAValue):
            indices = [indices]
        super().__init__(operands=(scalar, dest, indices), result_types=(dest.type,))

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.scalar)
        printer.print_string(" into ")
        printer.print_ssa_value(self.dest)
        printer.print_string("[")
        printer.print_list(self.indices, printer.print_ssa_value)
        printer.print_string("]")
        printer.print_string(" : ")
        printer.print_attribute(self.dest.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        scalar = parser.parse_operand()
        parser.parse_characters("into")
        dest = parser.parse_operand()
        indices = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.SQUARE, parse=parser.parse_operand
        )
        parser.parse_punctuation(":")
        parser.parse_type()
        return cls(scalar, dest, indices)


@irdl_op_definition
class FromElementsOp(IRDLOperation):
    """
    Tensor from elements operation.

    Create a N-D tensor from a range of same-type arguments. The number of provided
    elements should equal to the number of the elements in the result type.
    The elements correspond to a flattened tensor.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorfrom_elements-tensorfromelementsop
    """

    name = "tensor.from_elements"

    ELEMENT_TYPE: ClassVar = VarConstraint("ELEMENT_TYPE", AnyAttr())

    elements = var_operand_def(ELEMENT_TYPE)
    result = result_def(TensorType.constr(ELEMENT_TYPE))
    assembly_format = "$elements attr-dict `:` type($result)"


@irdl_op_definition
class SplatOp(IRDLOperation):
    """
    Tensor splat or broadcast operation.

    Broadcast the operand to all elements of the result tensor. An additional
    argument of type index must be provided for each dynamic dimension present
    in the result type.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorsplat-tensorsplatop
    """

    name = "tensor.splat"

    SPLAT_TYPE: ClassVar = VarConstraint("SPLAT_TYPE", AnyAttr())

    input = operand_def(SPLAT_TYPE)
    dynamicSizes = var_operand_def(IndexType)
    result = result_def(TensorType.constr(SPLAT_TYPE))
    assembly_format = "$input (`[` $dynamicSizes^ `]`)? attr-dict `:` type($result)"

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        input: SSAValue,
        dynamicSizes: Sequence[SSAValue | Operation],
        result_type: TensorType[Attribute],
    ):
        super().__init__(operands=(input, dynamicSizes), result_types=(result_type,))

    def verify_(self):
        if self.result.type.get_shape().count(DYNAMIC_INDEX) != len(self.dynamicSizes):
            raise VerifyException(
                "number of dynamic sizes must equal number of unknown dimensions in result tensor"
            )


@irdl_op_definition
class PadOp(IRDLOperation):
    """
    Tensor pad operation.

    tensor.pad is an operation that pads the source tensor with given low and high padding config.

    https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorpad-tensorpadop
    """

    name = "tensor.pad"

    source = operand_def(base(TensorType[Attribute]))
    low = var_operand_def(IndexType)
    high = var_operand_def(IndexType)
    static_low = prop_def(DenseArrayBase.constr(i64))
    static_high = prop_def(DenseArrayBase.constr(i64))
    nofold = opt_prop_def(UnitAttr)
    region = region_def("single_block")
    result = result_def(TensorType[Attribute])

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    assembly_format = (
        "$source "
        "(`nofold` $nofold^)? "
        "`low` `` custom<DynamicIndexList>($low, $static_low) "
        "`high` `` custom<DynamicIndexList>($high, $static_high) "
        "$region attr-dict `:` type($source) `to` type($result)"
    )

    custom_directives = (DynamicIndexList,)
    traits = traits_def(NoMemoryEffect(), AlwaysSpeculatable())

    def __init__(
        self,
        source: SSAValue | Operation,
        low: Sequence[SSAValue],
        high: Sequence[SSAValue],
        region: Region,
        static_low: Sequence[int] | DenseArrayBase,
        static_high: Sequence[int] | DenseArrayBase,
        nofold: UnitAttr,
        result_type: TensorType[Attribute],
        attributes: dict[str, Attribute] | None = None,
    ):
        if not isinstance(static_low, DenseArrayBase):
            static_low = DenseArrayBase.from_list(i64, static_low)

        if not isinstance(static_high, DenseArrayBase):
            static_high = DenseArrayBase.from_list(i64, static_high)

        super().__init__(
            operands=[source, low, high],
            result_types=[result_type],
            properties={
                "static_low": static_low,
                "static_high": static_high,
                "nofold": nofold,
            },
            attributes=attributes,
            regions=[region],
        )

    def verify_(self):
        if len(self.static_low) != len(self.static_high):
            raise VerifyException(
                f"pad sizes low ({len(self.static_low)}) and high ({len(self.static_high)})"
                " must have an equal number of dimensions"
            )
        source_type = self.source.type
        if isinstance(source_type, TensorType) and len(self.static_low) != len(
            source_type.get_shape()
        ):
            raise VerifyException(
                f"number of pad sizes ({len(self.static_low)}) must equal number of dimensions"
                f" in source tensor ({len(source_type.get_shape())})"
            )
        dynamic_dims = tuple(
            i
            for i, (l, h) in enumerate(
                zip(
                    self.static_low.get_values(),
                    self.static_high.get_values(),
                    strict=True,
                )
            )
            if l == DYNAMIC_INDEX or h == DYNAMIC_INDEX
        )
        result_dynamic_dims = tuple(
            i for i, s in enumerate(self.result.type.get_shape()) if s == DYNAMIC_INDEX
        )
        if len(result_dynamic_dims) != len(dynamic_dims):
            raise VerifyException(
                f"number of dynamic sizes ({len(dynamic_dims)})"
                f" must equal number of unknown dimensions in result tensor ({len(result_dynamic_dims)})"
            )
        if result_dynamic_dims != dynamic_dims:
            raise VerifyException(
                f"dynamic dimensions {dynamic_dims} don't correspond"
                f" with dynamic dimensions in the result tensor {result_dynamic_dims}"
            )
        if len(self.region.block.args) != len(self.static_low):
            raise VerifyException(
                "region must have an arg for each dimension of the source tensor"
                f" ({len(self.static_low)}) but region has ({len(self.region.block.args)})"
            )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "tensor.yield"

    traits = traits_def(IsTerminator())


Tensor = Dialect(
    "tensor",
    [
        CastOp,
        CollapseShapeOp,
        DimOp,
        EmptyOp,
        ExpandShapeOp,
        ExtractOp,
        ExtractSliceOp,
        FromElementsOp,
        InsertOp,
        InsertSliceOp,
        ReshapeOp,
        SplatOp,
        PadOp,
        YieldOp,
    ],
    [],
)
