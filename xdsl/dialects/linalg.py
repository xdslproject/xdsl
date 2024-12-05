from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from enum import auto
from typing import ClassVar, cast

from typing_extensions import Self

from xdsl.builder import Builder
from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    AnyMemRefType,
    AnyTensorType,
    ArrayAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
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
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
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


@irdl_op_definition
class GenericOp(IRDLOperation):
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
            printer.print(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        printer.print_string(" ")
        printer.print_region(self.body)

        if self.res:
            printer.print_string(" -> ")
            if len(self.res) == 1:
                printer.print_attribute(self.res[0].type)
            else:
                printer.print("(")
                printer.print_list(
                    self.res, lambda res: printer.print_attribute(res.type)
                )
                printer.print(")")

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


class NamedOpBase(IRDLOperation, ABC):
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
                res_types = [[parser.parse_attribute()]]
        else:
            res_types = ()

        return cls(ins, outs, res_types, attrs)

    def print(self, printer: Printer):
        extra_attrs = self.attributes.copy()
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
            printer.print(" attrs = ")
            printer.print_op_attributes(extra_attrs)

        if self.res:
            printer.print_string(" -> ")
            if len(self.res) == 1:
                printer.print_attribute(self.res[0].type)
            else:
                printer.print("(")
                printer.print_list(
                    self.res, lambda res: printer.print_attribute(res.type)
                )
                printer.print(")")

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
            if isa(op_type, MemRefType[Attribute]):
                element_type = op_type.get_element_type()
            elif isa(op_type, TensorType[Attribute]):
                element_type = op_type.get_element_type()
            else:  # int or float
                element_type = op_type
            assert isinstance(element_type, AnyFloat | IntegerType)
            result.append(element_type)

        return result


@irdl_op_definition
class AddOp(NamedOpBase):
    """
    Adds two tensors elementwise.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgadd-linalgaddop
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

        arg_types = self.body_arg_types((*inputs, *outputs))
        add = arith.AddfOp if isinstance(arg_types[-1], AnyFloat) else arith.AddiOp

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = add(args[0], args[1])
            YieldOp(result)

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=hidden_region,
        )


@irdl_op_definition
class SubOp(NamedOpBase):
    """
    Subtracts two tensors elementwise.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgsub-linalgsubop
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

        arg_types = self.body_arg_types((*inputs, *outputs))
        sub = arith.SubfOp if isinstance(arg_types[-1], AnyFloat) else arith.SubiOp

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = sub(args[0], args[1])
            YieldOp(result)

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=hidden_region,
        )


@irdl_op_definition
class FillOp(NamedOpBase):
    """
    Fills the output tensor with the given value.

    Works for arbitrary ranked output tensors since the operation performs scalar accesses
    only and is thus rank polymorphic. Numeric casting is performed on the value operand,
    promoting it to the same data type as the output.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgfill-linalgfillop
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

        arg_types = self.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            YieldOp(args[0])

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=hidden_region,
        )

    def verify_(self) -> None:
        # Check the that the inputs are of scalar type (f32, f64, etc)
        for value in self.inputs:
            if not isinstance(value.type, AnyFloat | IntegerType):
                raise VerifyException(
                    f"Input type is {value.type} but must be an instance of AnyFloat or IntegerType."
                )


@irdl_op_definition
class MulOp(NamedOpBase):
    """
    Multiplies two tensors elementwise.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmul-linalgmulop
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

        arg_types = self.body_arg_types((*inputs, *outputs))
        mul = arith.MulfOp if isinstance(arg_types[-1], AnyFloat) else arith.MuliOp

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = mul(args[0], args[1])
            YieldOp(result)

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=hidden_region,
        )


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Transpose operator

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgtranspose-linalgtransposeop
    """

    name = "linalg.transpose"

    input = operand_def(base(AnyMemRefType) | base(AnyTensorType))
    init = operand_def(base(AnyMemRefType) | base(AnyTensorType))
    result = var_result_def(AnyTensorType)

    permutation = attr_def(DenseArrayBase)

    def __init__(
        self,
        input: SSAValue,
        init: SSAValue,
        permutation: Attribute,
        result: Attribute | None = None,
    ):
        super().__init__(
            attributes={
                "permutation": permutation,
            },
            operands=(input, init),
            result_types=(result,),
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

        permutation_shape = cast(list[int], self.permutation.get_values())

        for i in range(len(input_shape)):
            input_dimension = input_shape[permutation_shape[i]]
            init_dimension = init_shape[i]

            if input_dimension != init_dimension:
                raise VerifyException(
                    f"dim(result, {i}) = {init_dimension} "
                    f"doesn't match dim(input, permutation[{i}]) = {input_dimension}"
                )

    def print(self, printer: Printer):
        printer.print_string(" ins(")
        printer.print(self.input)
        printer.print_string(":")
        printer.print(self.input.type)
        printer.print_string(")")
        printer.print_string(" outs(")
        printer.print(self.init)
        printer.print_string(":")
        printer.print(self.init.type)
        printer.print_string(") ")
        printer.print_string("permutation")
        printer.print_string(" = ")
        printer.print(list(self.permutation.get_values()))

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
        result = parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_keyword("permutation")
        parser.parse_punctuation("=")
        permutation = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        transpose = cls(
            input,
            init,
            DenseArrayBase.create_dense_int(i64, permutation),
            result,
        )
        return transpose


@irdl_op_definition
class MatmulOp(NamedOpBase):
    """
    Performs a matrix multiplication of two 2D inputs.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop

    """

    name = "linalg.matmul"

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

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

        arg_types = self.body_arg_types((*inputs, *outputs))
        add, mul = (
            (arith.AddfOp, arith.MulfOp)
            if isinstance(arg_types[-1], AnyFloat)
            else (arith.AddiOp, arith.MulfOp)
        )

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            result = mul(args[0], args[1])
            mac = add(result, args[2])
            YieldOp(mac)

        # add linalg.memoized_indexing_maps attribute
        if not attributes:
            attributes = {}
        if "linalg.memoized_indexing_maps" not in attributes:
            attributes["linalg.memoized_indexing_maps"] = ArrayAttr(
                [
                    AffineMapAttr(AffineMap.from_callable(lambda i, _, k: (i, k))),
                    AffineMapAttr(AffineMap.from_callable(lambda _, j, k: (k, j))),
                    AffineMapAttr(AffineMap.from_callable(lambda i, j, _: (i, j))),
                ]
            )

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=hidden_region,
        )


@irdl_op_definition
class QuantizedMatmulOp(NamedOpBase):
    """
    Performs a matrix multiplication of two 2D inputs.

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgquantized_matmul-linalgquantizedmatmulop

    """

    name = "linalg.quantized_matmul"

    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True

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

        arg_types = self.body_arg_types((*inputs, *outputs))

        @Builder.implicit_region(arg_types)
        def hidden_region(args: tuple[BlockArgument, ...]) -> None:
            o1 = arith.ExtSIOp(args[0], IntegerType(32))
            o2 = arith.SubiOp(o1, args[2])
            o3 = arith.ExtSIOp(args[1], IntegerType(32))
            o4 = arith.SubiOp(o3, args[3])
            o5 = arith.MuliOp(o2, o4)
            o6 = arith.AddiOp(args[4], o5)
            YieldOp(o6)

        # add linalg.memoized_indexing_maps attribute
        if not attributes:
            attributes = {}
        if "linalg.memoized_indexing_maps" not in attributes:
            attributes["linalg.memoized_indexing_maps"] = ArrayAttr(
                [
                    AffineMapAttr(AffineMap.from_callable(lambda i, _, k: (i, k))),
                    AffineMapAttr(AffineMap.from_callable(lambda _, j, k: (k, j))),
                    AffineMapAttr(AffineMap.from_callable(lambda i, j, _: (i, j))),
                ]
            )

        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=result_types,
            attributes=attributes,
            hidden_region=hidden_region,
        )


class PoolingOpsBase(IRDLOperation, ABC):
    """Base class for linalg pooling operations."""

    inputs = var_operand_def()
    outputs = var_operand_def(base(ShapedType))

    res = var_result_def(AnyTensorType)

    assembly_format = (
        "attr-dict `ins` `(` $inputs `:` type($inputs) `)` ` ` "
        "`outs` `(` $outputs `:` type($outputs) `)` `->` type($res)"
    )

    strides = attr_def(DenseIntOrFPElementsAttr)
    dilations = attr_def(DenseIntOrFPElementsAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        dilations: Attribute,
        strides: Attribute,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res
        super().__init__(
            attributes={
                "dilations": dilations,
                "strides": strides,
            },
            operands=(inputs, outputs),
            result_types=result_types,
        )


@irdl_op_definition
class PoolingNchwMaxOp(PoolingOpsBase):
    """
    Performs max pooling

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgpooling_nchw_max-linalgpoolingnchwmaxop
    """

    name = "linalg.pooling_nchw_max"


class ConvOpsBase(IRDLOperation, ABC):
    """Base class for linalg convolution operations."""

    inputs = var_operand_def()
    outputs = var_operand_def(base(ShapedType))

    res = var_result_def(AnyTensorType)

    assembly_format = (
        "attr-dict `ins` `(` $inputs `:` type($inputs) `)` ` ` "
        "`outs` `(` $outputs `:` type($outputs) `)` `->` type($res)"
    )

    strides = attr_def(DenseIntOrFPElementsAttr)
    dilations = attr_def(DenseIntOrFPElementsAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        dilations: Attribute,
        strides: Attribute,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue] = (),
        res: Sequence[Attribute] | None = None,
    ):
        if res is None:
            result_types = tuple(output.type for output in outputs)
        else:
            result_types = res
        super().__init__(
            attributes={
                "dilations": dilations,
                "strides": strides,
            },
            operands=(inputs, outputs),
            result_types=result_types,
        )


@irdl_op_definition
class Conv2DNchwFchwOp(ConvOpsBase):
    """
    Performs 2-D convolution

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nchw_fchw-linalgconv2dnchwfchwop
    """

    name = "linalg.conv_2d_nchw_fchw"


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    """
    Static broadcast operator

    Broadcast the input into the given shape by adding dimensions
    """

    name = "linalg.broadcast"

    input = operand_def(base(AnyMemRefType) | base(AnyTensorType))
    init = operand_def(base(AnyMemRefType) | base(AnyTensorType))
    result = var_result_def(AnyTensorType)

    dimensions = attr_def(DenseArrayBase)

    def __init__(
        self,
        input: SSAValue,
        init: SSAValue,
        dimensions: Attribute,
        result: Attribute | None = None,
    ):
        super().__init__(
            attributes={
                "dimensions": dimensions,
            },
            operands=(input, init),
            result_types=(result,),
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
        printer.print_string(" ins(")
        printer.print(self.input)
        printer.print_string(":")
        printer.print(self.input.type)
        printer.print_string(")")
        printer.print_string(" outs(")
        printer.print(self.init)
        printer.print_string(":")
        printer.print(self.init.type)
        printer.print_string(") ")
        printer.print_string("dimensions")
        printer.print_string(" = ")
        printer.print(list(self.dimensions.get_values()))

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
        result = parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_keyword("dimensions")
        parser.parse_punctuation("=")
        dimensions = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        broadcast = cls(
            input,
            init,
            DenseArrayBase.create_dense_int(i64, dimensions),
            result,
        )
        return broadcast


Linalg = Dialect(
    "linalg",
    [
        GenericOp,
        YieldOp,
        AddOp,
        SubOp,
        FillOp,
        MulOp,
        TransposeOp,
        MatmulOp,
        QuantizedMatmulOp,
        PoolingNchwMaxOp,
        Conv2DNchwFchwOp,
        BroadcastOp,
    ],
    [
        IteratorTypeAttr,
    ],
)
