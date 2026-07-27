from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar, NamedTuple, cast

from xdsl.builder import Builder
from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    ArrayAttr,
    DenseIntElementsAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    TensorType,
)
from xdsl.ir import Attribute, BlockArgument, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    prop_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.hints import isa

from .attrs import IteratorTypeAttr


class LoopBoundSource(NamedTuple):
    """Source information for one loop upper bound."""

    operand: SSAValue
    """The shaped operand that provides the bound."""

    dim_index: int
    """The dimension index in the operand used for the bound."""

    dim_size: int
    """The size of that dimension, or DYNAMIC_INDEX if it is dynamic."""


class LinalgStructuredOperation(IRDLOperation, ABC):
    """
    Abstract base class for structured linalg operations, allowing them to be processed
    via a unified interface.
    """

    inputs = var_operand_def()
    """
    The operands that won't be mutated.
    """
    outputs = var_operand_def(ShapedType)
    """
    The operands that will be accumulated into.
    These inputs may be `memref`s, which will be mutated in-place, or `tensor`s, which will be returned as results.
    """

    res = var_result_def(TensorType)
    """
    The updated `outputs`, empty if the inputs are memrefs.
    """

    body = region_def("single_block")
    """
    The body implementing the combination of scalar elements of the inputs, and
    yielding the scalar elements of the outputs.
    """

    @abstractmethod
    def get_indexing_maps(self) -> ArrayAttr[AffineMapAttr]:
        """
        Get the indexing maps corresponding to this operation's operands, in order.
        """

    @abstractmethod
    def get_iterator_types(self) -> ArrayAttr[IteratorTypeAttr]:
        """
        Get the iterator types corresponding to this operation's loop, in order.
        """

    def get_num_loops(self) -> int:
        return self.get_indexing_maps().data[0].data.num_dims

    def get_loops_to_shapes_map(self) -> AffineMap:
        """
        Returns a map to answer the question: "given an iteration space over
        the codomain, what are the subshapes of the operands involved in the
        computation".
        The default behavior is to just concatenate all the indexing maps.
        """
        indexing_maps = tuple(attr.data for attr in self.get_indexing_maps())
        result_exprs = tuple(res for map in indexing_maps for res in map.results)

        dims = self.get_num_loops()

        # FIXME: Support symbols.
        for map in indexing_maps:
            if map.num_symbols != 0:
                raise NotImplementedError(
                    "Indexing maps with symbols not supported for now."
                )

        syms = 0
        return AffineMap(dims, syms, result_exprs)

    def get_shapes_to_loops_map(self) -> AffineMap:
        """
        Returns a map to answer the question: "Given a list of operand ranges,
        what is the subportion of the iteration space involved in the
        computation". This is the inverse problem of `get_loops_to_shapes_map`.
        Return the empty AffineMap when such an AffineMap cannot be
        constructed. The default behavior is based on a very simple inference
        procedure that only works with permutation affine maps. A more advanced
        Tensor-Comprehension like inference is possible but has proven to be
        ambiguous in unfavorable case. A safer and more robust alternative is
        to allow each op to define its own AffineMap.
        """
        loops_to_shapes = self.get_loops_to_shapes_map()
        inverse = loops_to_shapes.inverse_permutation()
        if not inverse:
            raise NotImplementedError(
                "Non-invertible maps need dynamic shapes, which are not implemented."
            )
        return inverse

    def get_loop_bound_sources(
        self,
    ) -> tuple[LoopBoundSource, ...]:
        """
        Return where each loop upper bound comes from.

        Each entry identifies the shaped operand, the dimension index, and the size value.
        """
        shapes_to_loops = self.get_shapes_to_loops_map()

        needed_positions = tuple(
            expr.position
            for expr in shapes_to_loops.results
            if isinstance(expr, AffineDimExpr)
        )
        assert len(shapes_to_loops.results) == len(needed_positions)

        flat_shape_dims = tuple(
            LoopBoundSource(operand, dim_index, dim_size)
            for operand in self.operands
            if isa(operand, SSAValue[ShapedType])
            for dim_index, dim_size in enumerate(operand.type.get_shape())
        )

        return tuple(flat_shape_dims[position] for position in needed_positions)

    def get_static_shapes(self) -> list[int]:
        return [
            dim
            for operand in self.operands
            if isinstance(operand.type, ShapedType)
            for dim in operand.type.get_shape()
        ]

    def get_static_loop_ranges(self) -> tuple[int, ...]:
        shapes_to_loops = self.get_shapes_to_loops_map()
        static_shapes = self.get_static_shapes()
        return shapes_to_loops.eval(static_shapes, [])


class NamedOperation(LinalgStructuredOperation, ABC):
    """
    Abstract base class for named ops with hidden region.
    """

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = False

    def __init__(
        self,
        ins: Sequence[SSAValue],
        outs: Sequence[SSAValue],
        result_types: Sequence[Attribute | Sequence[Attribute] | None] | None = None,
        properties: Mapping[str, Attribute | None] | None = None,
        attributes: Mapping[str, Attribute | None] | None = None,
        hidden_region: Region | None = None,
    ):
        super().__init__(
            operands=[ins, outs],
            result_types=(
                result_types
                if result_types is not None and len(result_types) > 0
                else [[]]
            ),
            properties=properties,
            attributes=attributes,
            regions=[hidden_region],
        )

    @classmethod
    def parse(cls, parser: Parser):
        pos = parser.pos
        if cls.PRINT_ATTRS_IN_FRONT:
            attrs = parser.parse_optional_attr_dict()
        else:
            attrs = {}
        if parser.parse_optional_characters("ins"):
            parser.parse_punctuation("(")
            unresolved_ins = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            ins_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            ins = parser.resolve_operands(unresolved_ins, ins_types, pos)
        else:
            ins = ()

        pos = parser.pos
        if parser.parse_optional_characters("outs"):
            parser.parse_punctuation("(")
            unresolved_outs = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            outs_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            outs = parser.resolve_operands(unresolved_outs, outs_types, pos)
        else:
            outs = ()

        if not cls.PRINT_ATTRS_IN_FRONT:
            if parser.parse_optional_keyword("attrs"):
                parser.parse_punctuation("=")
                attrs = parser.expect(
                    parser.parse_optional_attr_dict, "expect extra attributes"
                )
            else:
                attrs = {}

        if parser.parse_optional_punctuation("->"):
            res_types = parser.parse_optional_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_attribute
            )
            if res_types is None:
                res_types = [parser.parse_attribute()]
        else:
            res_types = ()

        prop_names = cls.get_irdl_definition().properties

        properties = {k: v for k, v in attrs.items() if k in prop_names}
        # Drop the values in properties from attrs
        for k in properties:
            if k in attrs:
                del attrs[k]

        try:
            return cls.build(
                operands=(ins, outs),
                result_types=(res_types,),
                properties=properties,
                attributes=attrs,
                regions=(cls.get_hidden_region(ins, outs),),
            )
        except ValueError:
            parser.raise_error("Could not build linalg op")

    def print(self, printer: Printer):
        extra_attrs = {**self.attributes, **self.properties}
        extra_attrs.pop("indexing_maps", None)
        extra_attrs.pop("linalg.memoized_indexing_maps", None)
        extra_attrs.pop("iterator_types", None)
        extra_attrs.pop("doc", None)
        extra_attrs.pop("library_call", None)
        extra_attrs.pop("operandSegmentSizes", None)

        if extra_attrs and self.PRINT_ATTRS_IN_FRONT:
            printer.print_op_attributes(extra_attrs)
        if self.inputs:
            printer.print_string(" ins(")
            printer.print_list(self.inputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.inputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.outputs.types, printer.print_attribute)
            printer.print_string(")")

        if extra_attrs and not self.PRINT_ATTRS_IN_FRONT:
            printer.print_string(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        if self.res:
            printer.print_string(" -> ")
            if len(self.res) == 1:
                printer.print_attribute(self.res[0].type)
            else:
                with printer.in_parens():
                    printer.print_list(
                        self.res, lambda res: printer.print_attribute(res.type)
                    )

    @staticmethod
    def body_arg_types(
        operands: Sequence[SSAValue],
    ) -> Sequence[AnyFloat | IntegerType]:
        """
        Return the element types of the arguments of the body of this operation
        """

        result: Sequence[AnyFloat | IntegerType] = []

        for op in operands:
            op_type = op.type
            if isa(op_type, MemRefType | TensorType):
                element_type = op_type.get_element_type()
            else:  # int or float
                element_type = op_type
            assert isa(element_type, AnyFloat | IntegerType)
            result.append(element_type)

        return result

    @classmethod
    @abstractmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        """
        The hidden region for this linalg NamedOperation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_indexing_maps(self) -> Sequence[AffineMap]:
        """
        Get the default indexing maps corresponding to this operation's operands, in order.
        """

    def get_indexing_maps(self) -> ArrayAttr[AffineMapAttr]:
        return ArrayAttr(
            AffineMapAttr(map_) for map_ in self.get_default_indexing_maps()
        )


class ElementwiseOperation(NamedOperation, ABC):
    def get_default_indexing_maps(self) -> Sequence[AffineMap]:
        assert all(isinstance(t, ShapedType) for t in self.operand_types), (
            "Assume that all named linalg pointwise operations have matching shaped "
            "types."
        )
        operand_types = cast(Sequence[ShapedType], self.operand_types)
        shapes = tuple(t.get_shape() for t in operand_types)
        assert all(shape == shapes[0] for shape in shapes[1:]), (
            "All shapes must be equal"
        )

        return (AffineMap.identity(len(shapes[0])),) * len(operand_types)

    def get_iterator_types(self) -> ArrayAttr[IteratorTypeAttr]:
        num_loops = self.get_num_loops()
        return ArrayAttr((IteratorTypeAttr.parallel(),) * num_loops)


class PoolingOperation(NamedOperation, ABC):
    """Base class for linalg pooling operations."""

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

    strides = prop_def(DenseIntElementsAttr)
    dilations = prop_def(DenseIntElementsAttr)

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
        *,
        strides: DenseIntElementsAttr,
        dilations: DenseIntElementsAttr,
    ):
        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=res,
            attributes=attributes,
            properties={"strides": strides, "dilations": dilations},
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    def get_default_indexing_maps(self) -> Sequence[AffineMap]:
        raise NotImplementedError

    def get_iterator_types(self) -> ArrayAttr[IteratorTypeAttr]:
        raise NotImplementedError


class ConvOperation(NamedOperation, ABC):
    """Base class for linalg convolution operations."""

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

    strides = prop_def(DenseIntElementsAttr)
    dilations = prop_def(DenseIntElementsAttr)

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
        *,
        strides: DenseIntElementsAttr,
        dilations: DenseIntElementsAttr,
    ):
        super().__init__(
            ins=inputs,
            outs=outputs,
            attributes=attributes,
            result_types=res,
            properties={"strides": strides, "dilations": dilations},
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        from .ops import YieldOp

        arg_types = cls.body_arg_types((*inputs, *outputs))
        add, mul = (
            (arith.AddfOp, arith.MulfOp)
            if isinstance(arg_types[-1], AnyFloat)
            else (arith.AddiOp, arith.MuliOp)
        )

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            if arg_types[0] != arg_types[-1]:
                assert isinstance(arg_types[-1], IntegerType)
                a = arith.ExtSIOp(args[0], arg_types[-1])
                b = arith.ExtSIOp(args[1], arg_types[-1])
            else:
                a = args[0]
                b = args[1]
            result = mul(a, b)
            mac = add(result, args[2])
            YieldOp(mac)

        return hidden_region

    def get_default_indexing_maps(self) -> Sequence[AffineMap]:
        raise NotImplementedError

    def get_iterator_types(self) -> ArrayAttr[IteratorTypeAttr]:
        raise NotImplementedError
