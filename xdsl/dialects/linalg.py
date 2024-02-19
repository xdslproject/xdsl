from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    AnyShapedType,
    AnyTensorType,
    ArrayAttr,
    IntegerType,
    ShapedType,
    StringAttr,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
)
from xdsl.ir import Attribute, Data, Dialect, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    VarOperand,
    VarOpResult,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    prop_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException


class IteratorType(Enum):
    "Iterator type for linalg trait"

    PARALLEL = "parallel"
    REDUCTION = "reduction"
    WINDOW = "window"


@irdl_attr_definition
class IteratorTypeAttr(Data[IteratorType]):
    name = "linalg.iterator_type"

    @classmethod
    def parallel(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.PARALLEL)

    @classmethod
    def reduction(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.REDUCTION)

    @classmethod
    def window(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.WINDOW)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> IteratorType:
        with parser.in_angle_brackets():
            if parser.parse_optional_keyword("parallel") is not None:
                return IteratorType.PARALLEL
            if parser.parse_optional_keyword("reduction") is not None:
                return IteratorType.REDUCTION
            if parser.parse_optional_keyword("window") is not None:
                return IteratorType.WINDOW
            parser.raise_error("`parallel`, `reduction` or `window` expected")

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            data = self.data
            match data:
                case IteratorType.PARALLEL:
                    printer.print_string("parallel")
                case IteratorType.REDUCTION:
                    printer.print_string("reduction")
                case IteratorType.WINDOW:
                    printer.print_string("window")


@irdl_op_definition
class Generic(IRDLOperation):
    name = "linalg.generic"

    inputs: VarOperand = var_operand_def()
    outputs: VarOperand = var_operand_def(AnyShapedType())

    res: VarOpResult = var_result_def(AnyTensorType)

    body: Region = region_def("single_block")

    # Trait attributes
    indexing_maps: ArrayAttr[AffineMapAttr] = prop_def(ArrayAttr[AffineMapAttr])
    iterator_types: ArrayAttr[IteratorTypeAttr] = prop_def(ArrayAttr[IteratorTypeAttr])
    doc: StringAttr | None = opt_attr_def(StringAttr)
    library_call: StringAttr | None = opt_attr_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        indexing_maps: Sequence[AffineMapAttr] | ArrayAttr[AffineMapAttr],
        iterator_types: Sequence[Attribute] | ArrayAttr[Attribute],
        result_types: Sequence[Attribute] = (),
        doc: StringAttr | None = None,
        library_call: StringAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            result_types=[result_types],
            properties={
                "indexing_maps": ArrayAttr(indexing_maps),
                "iterator_types": ArrayAttr(iterator_types),
            },
            attributes={
                "doc": doc,
                "library_call": library_call,
            },
            regions=[body],
        )

    def get_indexing_maps(self) -> list[AffineMap]:
        return [attr.data for attr in self.indexing_maps]

    def get_num_loops(self) -> int:
        return self.indexing_maps.data[0].data.num_dims

    def get_loops_to_shapes_map(self) -> AffineMap:
        """
        Returns a map to answer the question: "given an iteration space over
        the codomain, what are the subshapes of the operands involved in the
        computation".
        The default behavior is to just concatenate all the indexing maps.
        """
        result_exprs = tuple(
            res for map in self.get_indexing_maps() for res in map.results
        )

        dims = self.get_num_loops()

        # FIXME: Support symbols.
        for map in self.get_indexing_maps():
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

    def print(self, printer: Printer):
        printer.print_string(" {indexing_maps = ")
        printer.print_attribute(self.indexing_maps)
        printer.print_string(", iterator_types = [")
        printer.print_list(
            self.iterator_types,
            lambda iterator_type: printer.print_string_literal(
                iterator_type.data.value
            ),
        )
        printer.print_string("]")
        if self.doc:
            printer.print_string(", doc = ")
            printer.print_attribute(self.doc)
        if self.library_call:
            printer.print_string(", library_call = ")
            printer.print_attribute(self.library_call)
        printer.print_string("}")

        if self.inputs:
            printer.print_string(" ins(")
            printer.print_list(self.inputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list((i.type for i in self.inputs), printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list((o.type for o in self.outputs), printer.print_attribute)
            printer.print_string(")")

        extra_attrs = self.attributes.copy()
        if "indexing_maps" in extra_attrs:
            del extra_attrs["indexing_maps"]
        if "iterator_types" in extra_attrs:
            del extra_attrs["iterator_types"]
        if "doc" in extra_attrs:
            del extra_attrs["doc"]
        if "library_call" in extra_attrs:
            del extra_attrs["library_call"]

        if extra_attrs:
            printer.print(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        printer.print_string(" ")
        printer.print_region(self.body)

        if self.res:
            printer.print_string(" -> ")
            printer.print_list(self.res, lambda res: printer.print_attribute(res.type))

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs_start_pos = parser.pos
        attrs = parser.parse_optional_attr_dict()
        attrs_end_pos = parser.pos

        if "indexing_maps" in attrs:
            indexing_maps = attrs["indexing_maps"]
            assert isinstance(indexing_maps, ArrayAttr)
            indexing_maps = cast(ArrayAttr[AffineMapAttr], indexing_maps)
            del attrs["indexing_maps"]
        else:
            parser.raise_error(
                "Expected indexing_maps for linalg.generic",
                attrs_start_pos,
                attrs_end_pos,
            )

        if "iterator_types" in attrs:
            # Get iterator types and make sure they're an ArrayAttr
            parsed_iterator_types = attrs["iterator_types"]
            assert isinstance(parsed_iterator_types, ArrayAttr)
            parsed_iterator_types = cast(ArrayAttr[Attribute], parsed_iterator_types)
            del attrs["iterator_types"]

            # Make sure they're iterator types
            iterator_types: list[IteratorTypeAttr] = []
            for iterator_type in parsed_iterator_types:
                match iterator_type:
                    case IteratorTypeAttr():
                        iterator_types.append(iterator_type)
                    case StringAttr():
                        iterator_type = IteratorTypeAttr(
                            IteratorType(iterator_type.data)
                        )
                        iterator_types.append(iterator_type)
                    case _:
                        parser.raise_error(
                            f"Unknown iterator type {iterator_type}",
                            attrs_start_pos,
                            attrs_end_pos,
                        )
        else:
            parser.raise_error(
                "Expected iterator_types for linalg.generic",
                attrs_start_pos,
                attrs_end_pos,
            )

        if "doc" in attrs:
            doc = attrs["doc"]
            assert isinstance(doc, StringAttr)
            del attrs["doc"]
        else:
            doc = None

        if "library_call" in attrs:
            library_call = attrs["library_call"]
            assert isinstance(library_call, StringAttr)
            del attrs["library_call"]
        else:
            library_call = None

        pos = parser.pos
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

        if parser.parse_optional_keyword("attrs"):
            parser.parse_punctuation("=")
            extra_attrs = parser.expect(
                parser.parse_optional_attr_dict, "expect extra attributes"
            )
        else:
            extra_attrs = {}

        body = parser.parse_region()

        if parser.parse_optional_punctuation("->"):
            res_types = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_attribute
            )
        else:
            res_types = ()

        generic = cls(
            ins,
            outs,
            body,
            indexing_maps,
            iterator_types,
            res_types,
            doc,
            library_call,
        )
        generic.attributes |= attrs
        generic.attributes |= extra_attrs

        return generic


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "linalg.yield"

    traits = frozenset([IsTerminator()])


@irdl_op_definition
class AddOp(IRDLOperation):
    """
    Adds two tensors elementwise.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgadd-linalgaddop
    """

    name = "linalg.add"

    inputs = var_operand_def()
    outputs = var_operand_def(AnyShapedType())

    res = var_result_def(AnyTensorType)

    assembly_format = (
        "`ins` `(` $inputs `:` type($inputs) `)` ` ` "
        "`outs` `(` $outputs `:` type($outputs) `)` `->` type($res) attr-dict"
    )

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res
        super().__init__(
            operands=(inputs, outputs),
            result_types=result_types,
        )


@irdl_op_definition
class FillOp(IRDLOperation):
    """
    Fills the output tensor with the given value.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgfill-linalgfillop
    """

    name = "linalg.fill"

    inputs = var_operand_def()
    outputs = var_operand_def(AnyShapedType())

    res = var_result_def(AnyTensorType)

    assembly_format = (
        "`ins` `(` $inputs `:` type($inputs) `)` ` ` "
        "`outs` `(` $outputs `:` type($outputs) `)` `->` type($res) attr-dict"
    )

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            operands=(inputs, outputs),
            result_types=result_types,
        )

    def verify_(self) -> None:
        # Check the that the inputs are of scalar type (f32, f64, etc)
        for value in self.inputs:
            if not isinstance(value.type, AnyFloat | IntegerType):
                raise VerifyException(
                    f"Input type is {value.type} but must be an instance of AnyFloat or IntegerType."
                )


Linalg = Dialect(
    "linalg",
    [
        Generic,
        YieldOp,
        AddOp,
        FillOp,
    ],
    [
        IteratorTypeAttr,
    ],
)
