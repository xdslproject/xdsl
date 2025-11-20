from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import auto
from typing import ClassVar, cast

from typing_extensions import Self

from xdsl.builder import Builder
from xdsl.dialects import arith, math
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    AnyTensorType,
    ArrayAttr,
    DenseArrayBase,
    DenseIntElementsAttr,
    IndexType,
    IndexTypeConstr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
)
from xdsl.ir import (
    Attribute,
    BlockArgument,
    Dialect,
    EnumAttribute,
    Region,
    SSAValue,
)
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    attr_def,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.str_enum import StrEnum


class IteratorType(StrEnum):
    "Iterator type for linalg trait"

    PARALLEL = auto()
    REDUCTION = auto()
    WINDOW = auto()


@irdl_attr_definition
class IteratorTypeAttr(EnumAttribute[IteratorType]):
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
            return super().parse_parameter(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)


class LinalgOperation(IRDLOperation, ABC):
    """
    Abstract base class for linalg operations, allowing them to be processed in with a
    unified interface.
    """

    @abstractmethod
    def get_indexing_maps(self) -> Sequence[AffineMap]:
        """
        Get the indexing maps corresponding to this operation's operands, in order.
        """

    def get_num_loops(self) -> int:
        return self.get_indexing_maps()[0].num_dims

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


@irdl_op_definition
class GenericOp(LinalgOperation):
    name = "linalg.generic"

    inputs = var_operand_def()
    outputs = var_operand_def(base(ShapedType))

    res = var_result_def(AnyTensorType)

    body = region_def("single_block")

    # Trait attributes
    indexing_maps = prop_def(ArrayAttr[AffineMapAttr])
    iterator_types = prop_def(ArrayAttr[IteratorTypeAttr])
    doc = opt_prop_def(StringAttr)
    library_call = opt_prop_def(StringAttr)

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
                "doc": doc,
                "library_call": library_call,
            },
            regions=[body],
        )

    def get_indexing_maps(self) -> Sequence[AffineMap]:
        return tuple(attr.data for attr in self.indexing_maps)

    def print(self, printer: Printer):
        printer.print_string(" {indexing_maps = ")
        printer.print_attribute(self.indexing_maps)
        printer.print_string(", iterator_types = [")
        printer.print_list(
            self.iterator_types,
            lambda iterator_type: printer.print_string_literal(iterator_type.data),
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
            printer.print_list(self.inputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.outputs.types, printer.print_attribute)
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
            printer.print_string(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        printer.print_string(" ")
        printer.print_region(self.body)

        if self.res:
            printer.print_string(" -> ")
            if len(self.res) == 1:
                printer.print_attribute(self.res[0].type)
            else:
                with printer.in_parens():
                    printer.print_list(
                        self.res, lambda res: printer.print_attribute(res.type)
                    )

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
        generic.attributes |= extra_attrs

        return generic


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "linalg.yield"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class IndexOp(IRDLOperation):
    name = "linalg.index"

    dim = prop_def(IntegerAttr[i64])

    result = result_def(IndexTypeConstr)

    traits = traits_def(HasParent(GenericOp))

    assembly_format = "$dim attr-dict `:` type($result)"

    def __init__(
        self,
        dim: int,
    ):
        dim_attr = IntegerAttr(dim, i64)
        super().__init__(properties={"dim": dim_attr}, result_types=[IndexType()])


class NamedOperation(IRDLOperation, ABC):
    """
    Abstract base class for named ops with hidden region.
    """

    inputs = var_operand_def()
    outputs = var_operand_def(base(ShapedType))

    res = var_result_def(AnyTensorType)

    hidden_region = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

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
        if "indexing_maps" in extra_attrs:
            del extra_attrs["indexing_maps"]
        if "linalg.memoized_indexing_maps" in extra_attrs:
            del extra_attrs["linalg.memoized_indexing_maps"]
        if "iterator_types" in extra_attrs:
            del extra_attrs["iterator_types"]
        if "doc" in extra_attrs:
            del extra_attrs["doc"]
        if "library_call" in extra_attrs:
            del extra_attrs["library_call"]
        if "operandSegmentSizes" in extra_attrs:
            del extra_attrs["operandSegmentSizes"]

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


@irdl_op_definition
class AddOp(NamedOperation):
    """
    Adds two tensors elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgadd-linalgaddop).
    """

    name = "linalg.add"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        add = arith.AddfOp if isinstance(arg_types[-1], AnyFloat) else arith.AddiOp

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = add(args[0], args[1])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class ExpOp(NamedOperation):
    """
    Applies exp(x) elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgexp-linalgexpop).
    """

    name = "linalg.exp"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = math.ExpOp(args[0])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class LogOp(NamedOperation):
    """
    Applies log(x) elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalglog-linalglogop).
    """

    name = "linalg.log"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = math.LogOp(args[0])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class SubOp(NamedOperation):
    """
    Subtracts two tensors elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgsub-linalgsubop).
    """

    name = "linalg.sub"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        self.body_arg_types((*inputs, *outputs))

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        sub = arith.SubfOp if isinstance(arg_types[-1], AnyFloat) else arith.SubiOp

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = sub(args[0], args[1])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class SqrtOp(NamedOperation):
    """
    Applies sqrt(x) elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgsqrt-linalgsqrtop).
    """

    name = "linalg.sqrt"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = math.SqrtOp(args[0])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class SelectOp(NamedOperation):
    """
    Chooses one value based on a binary condition supplied as its first operand.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgselect-linalgselectop).
    """

    name = "linalg.select"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = arith.SelectOp(*args[: len(inputs)])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class FillOp(NamedOperation):
    """
    Fills the output tensor with the given value.

    Works for arbitrary ranked output tensors since the operation performs scalar accesses
    only and is thus rank polymorphic. Numeric casting is performed on the value operand,
    promoting it to the same data type as the output.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgfill-linalgfillop).
    """

    name = "linalg.fill"

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            assert isa(outputs, Sequence[SSAValue]), "cannot infer result_types"
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    def verify_(self) -> None:
        # Check the that the inputs are of scalar type (f32, f64, etc)
        for value in self.inputs:
            if not isinstance(value.type, AnyFloat | IntegerType):
                raise VerifyException(
                    f"Input type is {value.type} but must be an instance of AnyFloat or IntegerType."
                )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            YieldOp(args[0])

        return hidden_region


@irdl_op_definition
class CopyOp(NamedOperation):
    """
    Copies the tensor elementwise.

    Numeric casting is performed on the input operand, promoting it to the same data type as the accumulator/output.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgcopy-linalgcopyop).
    """

    name = "linalg.copy"

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            assert isa(outputs, Sequence[SSAValue]), "cannot infer result_types"
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            YieldOp(args[0])

        return hidden_region


@irdl_op_definition
class MaxOp(NamedOperation):
    """
    Takes the max (signed) between two inputs, elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmax-linalgmaxop).
    """

    name = "linalg.max"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        maxop = (
            arith.MaximumfOp if isinstance(arg_types[-1], AnyFloat) else arith.MaxSIOp
        )

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = maxop(args[0], args[1])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class MinOp(NamedOperation):
    """
    Takes the max (signed) between two inputs, elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmax-linalgmaxop).
    """

    name = "linalg.min"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        minop = (
            arith.MinimumfOp if isinstance(arg_types[-1], AnyFloat) else arith.MinSIOp
        )

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = minop(args[0], args[1])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class MulOp(NamedOperation):
    """
    Multiplies two tensors elementwise.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmul-linalgmulop).
    """

    name = "linalg.mul"

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        mul = arith.MulfOp if isinstance(arg_types[-1], AnyFloat) else arith.MuliOp

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = mul(args[0], args[1])
            YieldOp(result)

        return hidden_region


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Transpose operator

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgtranspose-linalgtransposeop).
    """

    name = "linalg.transpose"

    input = operand_def(base(MemRefType) | base(AnyTensorType))
    init = operand_def(base(MemRefType) | base(AnyTensorType))
    result = var_result_def(AnyTensorType)

    hidden_region = region_def("single_block")

    permutation = prop_def(DenseArrayBase.constr(i64))

    def __init__(
        self,
        input: SSAValue,
        init: SSAValue,
        permutation: Attribute,
        result: Attribute | None = None,
    ):
        if result is None:
            if isa(init.type, TensorType):
                results = (init.type,)
            else:
                results = ()
        else:
            results = (result,)

        arg_types = NamedOperation.body_arg_types((input, init))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            YieldOp(args[0])

        super().__init__(
            properties={
                "permutation": permutation,
            },
            operands=(input, init),
            result_types=(results,),
            regions=(hidden_region,),
        )

    def verify_(self) -> None:
        assert isinstance(input_type := self.input.type, TensorType | MemRefType)
        assert isinstance(init_type := self.init.type, TensorType | MemRefType)

        input_shape = input_type.get_shape()
        init_shape = init_type.get_shape()

        if (input_rank := len(input_shape)) != (init_rank := len(init_shape)):
            raise VerifyException(
                f"Input rank ({input_rank}) does not match output rank ({init_rank})"
            )
        if (input_rank := len(input_shape)) != (
            permutation_size := len(self.permutation)
        ):
            raise VerifyException(
                f"Input rank ({input_rank}) does not match size of permutation ({permutation_size})"
            )

        permutation_shape = self.permutation.get_values()

        for i in range(len(input_shape)):
            input_dimension = input_shape[permutation_shape[i]]
            init_dimension = init_shape[i]

            if input_dimension != init_dimension:
                raise VerifyException(
                    f"dim(result, {i}) = {init_dimension} "
                    f"doesn't match dim(input, permutation[{i}]) = {input_dimension}"
                )

    def print(self, printer: Printer):
        printer.print_string(" ins")
        with printer.in_parens():
            printer.print_ssa_value(self.input)
            printer.print_string(":")
            printer.print_attribute(self.input.type)
        printer.print_string(" outs")
        with printer.in_parens():
            printer.print_ssa_value(self.init)
            printer.print_string(":")
            printer.print_attribute(self.init.type)
        printer.print_string(" permutation = ")
        with printer.in_square_brackets():
            printer.print_list(self.permutation.get_values(), printer.print_int)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_characters("ins")
        parser.parse_punctuation("(")
        input = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_characters("outs")
        parser.parse_punctuation("(")
        init = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_keyword("permutation")
        parser.parse_punctuation("=")
        permutation = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        transpose = cls(
            input,
            init,
            DenseArrayBase.from_list(i64, permutation),
        )
        return transpose


@irdl_op_definition
class MatmulOp(NamedOperation):
    """
    Performs a matrix multiplication of two 2D inputs.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop).
    """

    name = "linalg.matmul"

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

    indexing_maps = prop_def(
        ArrayAttr[AffineMapAttr],
        default_value=ArrayAttr(
            [
                AffineMapAttr(AffineMap.from_callable(lambda i, _, k: (i, k))),
                AffineMapAttr(AffineMap.from_callable(lambda _, j, k: (k, j))),
                AffineMapAttr(AffineMap.from_callable(lambda i, j, _: (i, j))),
            ]
        ),
    )

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(
                cast(AnyTensorType, output_type)
                for output in outputs
                if isinstance(output_type := output.type, TensorType)
            )
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        add, mul = (
            (arith.AddfOp, arith.MulfOp)
            if isinstance(arg_types[-1], AnyFloat)
            else (arith.AddiOp, arith.MuliOp)
        )

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = mul(args[0], args[1])
            mac = add(result, args[2])
            YieldOp(mac)

        return hidden_region


@irdl_op_definition
class QuantizedMatmulOp(NamedOperation):
    """
    Performs a matrix multiplication of two 2D inputs.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgquantized_matmul-linalgquantizedmatmulop).
    """

    name = "linalg.quantized_matmul"

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

    memoized_indexing_maps = attr_def(
        ArrayAttr[AffineMapAttr],
        default_value=ArrayAttr(
            [
                AffineMapAttr(AffineMap.from_callable(lambda i, _, k: (i, k))),
                AffineMapAttr(AffineMap.from_callable(lambda _, j, k: (k, j))),
                AffineMapAttr(AffineMap(3, 0, ())),
                AffineMapAttr(AffineMap(3, 0, ())),
                AffineMapAttr(AffineMap.from_callable(lambda i, j, _: (i, j))),
            ]
        ),
        attr_name="linalg.memoized_indexing_maps",
    )

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(
                cast(AnyTensorType, output_type)
                for output in outputs
                if isinstance(output_type := output.type, TensorType)
            )
        else:
            result_types = res

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=self.get_hidden_region(inputs, outputs),
        )

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            o1 = arith.ExtSIOp(args[0], IntegerType(32))
            o2 = arith.SubiOp(o1, args[2])
            o3 = arith.ExtSIOp(args[1], IntegerType(32))
            o4 = arith.SubiOp(o3, args[3])
            o5 = arith.MuliOp(o2, o4)
            o6 = arith.AddiOp(args[4], o5)
            YieldOp(o6)

        return hidden_region


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


@irdl_op_definition
class PoolingNchwMaxOp(PoolingOperation):
    """
    Performs max pooling

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgpooling_nchw_max-linalgpoolingnchwmaxop).
    """

    name = "linalg.pooling_nchw_max"

    @classmethod
    def get_hidden_region(
        cls, inputs: Sequence[SSAValue], outputs: Sequence[SSAValue]
    ) -> Region:
        arg_types = cls.body_arg_types((*inputs, *outputs))
        max_op = (
            arith.MaximumfOp if isinstance(arg_types[-1], AnyFloat) else arith.MaxSIOp
        )

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = max_op(args[0], args[1])
            YieldOp(result)

        return hidden_region


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


@irdl_op_definition
class Conv2DNchwFchwOp(ConvOperation):
    """
    Performs 2-D convolution

    See external [documentation](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nchw_fchw-linalgconv2dnchwfchwop).
    """

    name = "linalg.conv_2d_nchw_fchw"


@irdl_op_definition
class Conv2DNgchwFgchwOp(ConvOperation):
    name = "linalg.conv_2d_ngchw_fgchw"


@irdl_op_definition
class Conv2DNgchwGfchwOp(ConvOperation):
    name = "linalg.conv_2d_ngchw_gfchw"


@irdl_op_definition
class Conv2DNhwc_FhwcOp(ConvOperation):
    name = "linalg.conv_2d_nhwc_fhwc"


@irdl_op_definition
class Conv2DNhwc_HwcfOp(ConvOperation):
    name = "linalg.conv_2d_nhwc_hwcf"


@irdl_op_definition
class Conv2DNhwgcGfhwcOp(ConvOperation):
    name = "linalg.conv_2d_nhwgc_gfhwc"


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    """
    Static broadcast operator

    Broadcast the input into the given shape by adding dimensions
    """

    name = "linalg.broadcast"

    input = operand_def(base(MemRefType) | base(AnyTensorType))
    init = operand_def(base(MemRefType) | base(AnyTensorType))
    result = var_result_def(AnyTensorType)

    hidden_region = region_def("single_block")

    dimensions = prop_def(DenseArrayBase.constr(i64))

    def __init__(
        self,
        input: SSAValue,
        init: SSAValue,
        dimensions: Attribute,
        result: Attribute | None = None,
    ):
        if result is None:
            if isa(init.type, TensorType):
                results = (init.type,)
            else:
                results = ()
        else:
            results = (result,)

        arg_types = NamedOperation.body_arg_types((input, init))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            YieldOp(args[0])

        super().__init__(
            properties={
                "dimensions": dimensions,
            },
            operands=(input, init),
            result_types=(results,),
            regions=(hidden_region,),
        )

    def verify_(self) -> None:
        assert isinstance(input_type := self.input.type, TensorType | MemRefType)
        assert isinstance(init_type := self.init.type, TensorType | MemRefType)

        dimensions_shape = self.dimensions.get_values()

        input_shape = input_type.get_shape()
        init_shape = init_type.get_shape()

        if (input_and_dims_rank := (len(input_shape) + len(dimensions_shape))) != (
            init_rank := len(init_shape)
        ):
            raise VerifyException(
                f"Input rank plus added dimensions ({input_and_dims_rank}) does not match output rank ({init_rank})"
            )

        for index, dim in enumerate(dimensions_shape):
            if dim < 0 or dim >= init_rank:
                raise VerifyException(
                    f"Dimension {index} is out of range.  Expected range: [0, {init_rank - 1}], got: {dim}"
                )

        # intialise an array to store the dimensions being mapped
        dimensions_map: list[int] = []
        for dim in range(init_rank):
            if dim not in dimensions_shape:
                dimensions_map.append(dim)

        for input_dim_index, init_dim_index in enumerate(dimensions_map):
            if input_shape[input_dim_index] != init_shape[init_dim_index]:
                raise VerifyException(
                    f"input dimension {input_dim_index} should match output dimension {init_dim_index}. "
                    f"input: {input_shape[input_dim_index]}, output: {init_shape[init_dim_index]}"
                )

    def print(self, printer: Printer):
        printer.print_string(" ins")
        with printer.in_parens():
            printer.print_ssa_value(self.input)
            printer.print_string(":")
            printer.print_attribute(self.input.type)
        printer.print_string(" outs")
        with printer.in_parens():
            printer.print_ssa_value(self.init)
            printer.print_string(":")
            printer.print_attribute(self.init.type)
        printer.print_string(" dimensions = ")
        with printer.in_square_brackets():
            printer.print_list(self.dimensions.get_values(), printer.print_int)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_characters("ins")
        parser.parse_punctuation("(")
        input = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_characters("outs")
        parser.parse_punctuation("(")
        init = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_keyword("dimensions")
        parser.parse_punctuation("=")
        dimensions = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        broadcast = cls(
            input,
            init,
            DenseArrayBase.from_list(i64, dimensions),
        )
        return broadcast


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "linalg.reduce"

    input = operand_def(base(MemRefType) | base(AnyTensorType))
    init = operand_def(base(MemRefType) | base(AnyTensorType))
    result = var_result_def(AnyTensorType)

    region: Region = region_def("single_block")

    dimensions = prop_def(DenseArrayBase.constr(i64))

    def __init__(
        self,
        input: SSAValue,
        init: SSAValue,
        dimensions: Attribute,
        region: Region,
    ):
        if isa(init.type, TensorType):
            result = (init.type,)
        else:
            result = ()

        super().__init__(
            properties={
                "dimensions": dimensions,
            },
            operands=(input, init),
            regions=[region],
            result_types=[result],
        )

    def verify_(self) -> None:
        assert isinstance(input_type := self.input.type, TensorType | MemRefType)
        assert isinstance(init_type := self.init.type, TensorType | MemRefType)

        if input_type.get_element_type() != init_type.get_element_type():
            raise VerifyException(
                f"Reduction element types must be equal, but input is {input_type.get_element_type()} "
                f"and init is {init_type.get_element_type()}"
            )

        dimensions_shape = self.dimensions.get_values()
        input_shape = input_type.get_shape()
        init_shape = init_type.get_shape()

        if len(init_shape) != len(input_shape) - len(dimensions_shape):
            raise VerifyException(
                "Output rank must equal input rank minus number of dimensions being reduced over"
            )

        init_index = 0
        for input_index in range(len(input_shape)):
            if input_index not in dimensions_shape:
                if input_shape[input_index] != init_shape[init_index]:
                    raise VerifyException(
                        f"Non-reduced input dimension {input_index} must equal output dimension {init_index}"
                    )
                init_index += 1

    def print(self, printer: Printer):
        printer.print_string(" ins")
        with printer.in_parens():
            printer.print_ssa_value(self.input)
            printer.print_string(":")
            printer.print_attribute(self.input.type)
        printer.print_string(" outs")
        with printer.in_parens():
            printer.print_ssa_value(self.init)
            printer.print_string(":")
            printer.print_attribute(self.init.type)
        printer.print_string(" dimensions = ")
        with printer.in_square_brackets():
            printer.print_list(self.dimensions.get_values(), printer.print_int)
        printer.print_string("\n")
        with printer.in_parens():
            printer.print_list(self.region.blocks[0].args, printer.print_block_argument)
        printer.print_string(" ")
        printer.print_region(self.region, print_entry_block_args=False)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_characters("ins")
        parser.parse_punctuation("(")
        input = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_characters("outs")
        parser.parse_punctuation("(")
        init = parser.parse_operand()
        parser.parse_punctuation(":")
        parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_keyword("dimensions")
        parser.parse_punctuation("=")
        dimensions = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        entry_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_argument
        )
        region = parser.parse_region(entry_args)
        reduction = cls(
            input,
            init,
            DenseArrayBase.from_list(i64, dimensions),
            region,
        )
        return reduction


Linalg = Dialect(
    "linalg",
    [
        GenericOp,
        YieldOp,
        IndexOp,
        AddOp,
        ExpOp,
        LogOp,
        SubOp,
        SqrtOp,
        SelectOp,
        FillOp,
        CopyOp,
        MaxOp,
        MinOp,
        MulOp,
        TransposeOp,
        MatmulOp,
        QuantizedMatmulOp,
        PoolingNchwMaxOp,
        Conv2DNchwFchwOp,
        Conv2DNhwgcGfhwcOp,
        Conv2DNhwc_HwcfOp,
        Conv2DNgchwGfchwOp,
        Conv2DNgchwFgchwOp,
        Conv2DNhwc_FhwcOp,
        BroadcastOp,
        ReduceOp,
    ],
    [
        IteratorTypeAttr,
    ],
)
