from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from typing_extensions import Self

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    Annotated,
    AnySignlessIntegerOrIndexType,
    ArrayAttr,
    ContainerType,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    TensorType,
    UnrankedTensorType,
    i64,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    base,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import NoMemoryEffect
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    Convert a tensor from one type to an equivalent type without changing any data elements.
    The source and destination types must both be tensor types with the same element type.
    If both are ranked, then the rank should be the same and static dimensions should match.
    The operation is invalid if converting to a mismatching constant dimension.
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
    The tensor.dim operation takes a tensor and a dimension operand of type index.
    It returns the size of the requested dimension of the given tensor.
    If the dimension index is out of bounds, the behavior is undefined
    """

    name = "tensor.dim"

    source = operand_def(
        base(TensorType[Attribute]) | base(UnrankedTensorType[Attribute])
    )
    index = operand_def(IndexType)
    result = result_def(IndexType)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        source: SSAValue | Operation,
        index: SSAValue | Operation,
        attributes: Mapping[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=(source, index), result_types=(IndexType(),), attributes=attributes
        )

    def print(self, printer: Printer):
        printer.print_op_attributes(self.attributes)
        printer.print_string(" ")
        printer.print_ssa_value(self.source)
        printer.print_string(", ")
        printer.print_ssa_value(self.index)
        printer.print_string(" : ")
        printer.print_attribute(self.source.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attributes = parser.parse_optional_attr_dict()
        source = parser.parse_operand()
        parser.parse_punctuation(",")
        index = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        return cls(source, index, attributes)

    def verify_(self):
        if isinstance((source_type := self.source.type), TensorType):
            if not len(source_type.get_shape()):
                raise VerifyException("cannot get dim of 0-rank tensor")


@irdl_op_definition
class EmptyOp(IRDLOperation):
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


ReassociationAttr = ArrayAttr[
    ArrayAttr[IntegerAttr[Annotated[IntegerType, IntegerType(64)]]]
]


@irdl_op_definition
class CollapseShapeOp(IRDLOperation):
    name = "tensor.collapse_shape"

    src = operand_def(TensorType[Attribute])
    result = result_def(TensorType[Attribute])
    reassociation = prop_def(ReassociationAttr)
    assembly_format = (
        "$src $reassociation attr-dict `:` type($src) `into` type($result)"
    )

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    name = "tensor.reshape"

    source = operand_def(TensorType[Attribute])
    shape = operand_def(TensorType[AnySignlessIntegerOrIndexType])
    result = result_def(TensorType[Attribute])

    traits = traits_def(NoMemoryEffect())

    def __init__(self, source: SSAValue, shape: SSAValue, result_type: Attribute):
        super().__init__(
            operands=(
                source,
                shape,
            ),
            result_types=(result_type,),
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.source)
        printer.print_string("(")
        printer.print_ssa_value(self.shape)
        printer.print_string(")")
        printer.print_string(" : ")
        printer.print_string("(")
        printer.print_attribute(self.source.type)
        printer.print_string(", ")
        printer.print_attribute(self.shape.type)
        printer.print_string(")")
        printer.print_string(" -> ")
        printer.print_attribute(self.result.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs = parser.parse_optional_attr_dict()
        source = parser.parse_operand()
        parser.parse_punctuation("(")
        shape = parser.parse_operand()
        parser.parse_punctuation(")")
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        parser.parse_comma_separated_list(Parser.Delimiter.NONE, parser.parse_type)
        parser.parse_punctuation(")")
        parser.parse_optional_punctuation("->")
        result_type = parser.parse_attribute()

        reshape = cls(
            source,
            shape,
            result_type,
        )
        reshape.attributes |= attrs
        return reshape

    def verify_(self) -> None:
        if (
            not isinstance(source_type := self.source.type, TensorType)
            or not isinstance(shape_type := self.shape.type, TensorType)
            or not isinstance(res_type := self.result.type, TensorType)
        ):
            raise ValueError(
                "tensor elementwise operation operands and result must be of type TensorType"
            )

        source_type = cast(TensorType[Attribute], source_type)
        shape_type = cast(TensorType[Attribute], shape_type)
        res_type = cast(TensorType[Attribute], res_type)

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
class ExtractSliceOp(IRDLOperation):
    name = "tensor.extract_slice"

    source = operand_def(TensorType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    static_offsets = prop_def(DenseArrayBase)
    static_sizes = prop_def(DenseArrayBase)
    static_strides = prop_def(DenseArrayBase)
    result = result_def(TensorType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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
        source_v = SSAValue.get(source)
        source_t = source_v.type
        if not isinstance(source_t, ContainerType):
            raise ValueError(f"Expected ContainerType, got {source_t}")

        if reduce_rank:
            result_sizes = list(s for s in sizes if s != 1)
        else:
            result_sizes = list(sizes)

        return_type = TensorType[Any](source_t.get_element_type(), result_sizes)

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
    name = "tensor.insert_slice"

    source = operand_def(TensorType)
    dest = operand_def(TensorType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    static_offsets = prop_def(DenseArrayBase)
    static_sizes = prop_def(DenseArrayBase)
    static_strides = prop_def(DenseArrayBase)
    result = result_def(TensorType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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
            static_offsets = [memref.SubviewOp.DYNAMIC_INDEX] * len(offsets) + (
                [0] * (dims - len(offsets))
            )
        if not static_strides:
            static_strides = [memref.SubviewOp.DYNAMIC_INDEX] * len(strides) + (
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


Tensor = Dialect(
    "tensor",
    [
        CastOp,
        DimOp,
        EmptyOp,
        ExtractSliceOp,
        InsertSliceOp,
        ReshapeOp,
        CollapseShapeOp,
        ExtractOp,
        InsertOp,
    ],
    [],
)
