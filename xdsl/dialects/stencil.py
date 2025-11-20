from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import prod
from operator import add, lt, neg
from typing import Generic, TypeAlias, cast

from typing_extensions import TypeVar

from xdsl.dialects import builtin, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    AttrSizedOperandSegments,
    BaseAttr,
    ConstraintContext,
    IRDLOperation,
    MessageConstraint,
    ParamAttrConstraint,
    VarConstraint,
    attr_def,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    EffectInstance,
    HasAncestor,
    HasCanonicalizationPatternsTrait,
    HasParent,
    HasShapeInferencePatternsTrait,
    IsolatedFromAbove,
    IsTerminator,
    MemoryEffect,
    MemoryEffectKind,
    NoMemoryEffect,
    Pure,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

_FieldTypeElement = TypeVar(
    "_FieldTypeElement", bound=Attribute, covariant=True, default=Attribute
)


@irdl_attr_definition
class IndexAttr(ParametrizedAttribute, Iterable[int]):
    name = "stencil.index"

    array: ArrayAttr[IntAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            return [cls.parse_indices(parser)]

    @classmethod
    def parse_indices(cls, parser: AttrParser) -> ArrayAttr[IntAttr]:
        """
        Parse a comma-separated, square delimited, list of integers into an ArrayAttr of
        IntAttrs.

        e.g.: `[1, 2, 3]`
        """
        ints = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: parser.parse_integer(allow_boolean=False)
        )
        return ArrayAttr(IntAttr(i) for i in ints)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.print_indices(printer)

    def print_indices(self, printer: Printer) -> None:
        printer.print_string(f"[{', '.join(str(e) for e in self)}]")

    def verify(self) -> None:
        l = len(self)
        if l < 1 or l > 3:
            raise VerifyException(
                f"Expected 1 to 3 indexes for stencil.index, got {l}."
            )

    @staticmethod
    def get(*indices: int | IntAttr):
        return IndexAttr(
            ArrayAttr(
                [(IntAttr(idx) if isinstance(idx, int) else idx) for idx in indices]
            )
        )

    # TODO : come to an agreement on, do we want to allow that kind of things
    # on Attributes? Author's opinion is a clear yes :P
    def __neg__(self) -> IndexAttr:
        return IndexAttr.get(*(map(neg, self)))

    def __add__(self, o: IndexAttr) -> IndexAttr:
        return IndexAttr.get(*(map(add, self, o)))

    def __sub__(self, o: IndexAttr) -> IndexAttr:
        return self + -o

    def __lt__(self, o: IndexAttr) -> bool:
        return any(map(lt, self, o))

    @staticmethod
    def min(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        return IndexAttr.get(*map(min, a, b))

    @staticmethod
    def max(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        return IndexAttr.get(*map(max, a, b))

    def __len__(self):
        return len(self.array)

    def __iter__(self) -> Iterator[int]:
        return (e.data for e in self.array.data)


@irdl_attr_definition
class StencilBoundsAttr(ParametrizedAttribute):
    """
    This attribute represents known bounds over a stencil type.
    """

    name = "stencil.bounds"
    lb: IndexAttr
    ub: IndexAttr

    def _verify(self):
        if len(self.lb) != len(self.ub):
            raise VerifyException(
                "Incoherent stencil bounds: lower and upper bounds must have the "
                "same dimensionality."
            )
        for d in self.ub - self.lb:
            if d <= 0:
                raise VerifyException(
                    "Incoherent stencil bounds: upper bound must be strictly "
                    "greater than lower bound."
                )

    def __init__(self, bounds: Iterable[tuple[int | IntAttr, int | IntAttr]]):
        if bounds:
            lb, ub = zip(*bounds)
        else:
            lb, ub = (), ()
        super().__init__(
            IndexAttr.get(*lb),
            IndexAttr.get(*ub),
        )

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.lb.print_indices(printer)
            printer.print_string(", ")
            self.ub.print_indices(printer)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            lb = IndexAttr(IndexAttr.parse_indices(parser))
            parser.parse_punctuation(",")
            ub = IndexAttr(IndexAttr.parse_indices(parser))
            return [lb, ub]

    def union(self, other: StencilBoundsAttr | IntAttr) -> StencilBoundsAttr:
        if isinstance(other, IntAttr):
            return self
        return StencilBoundsAttr(
            zip(
                map(min, self.lb, other.lb),
                map(max, self.ub, other.ub),
            )
        )

    def intersection(self, other: StencilBoundsAttr | IntAttr) -> StencilBoundsAttr:
        if isinstance(other, IntAttr):
            return self
        return StencilBoundsAttr(
            zip(
                map(max, self.lb, other.lb),
                map(min, self.ub, other.ub),
            )
        )

    def __or__(self, value: StencilBoundsAttr | IntAttr) -> StencilBoundsAttr:
        return self.union(value)

    def __ror__(self, value: StencilBoundsAttr | IntAttr) -> StencilBoundsAttr:
        return self | value

    def __and__(self, value: StencilBoundsAttr | IntAttr) -> StencilBoundsAttr:
        return self.intersection(value)

    def __rand__(self, value: StencilBoundsAttr | IntAttr) -> StencilBoundsAttr:
        return self & value

    def __add__(self, o: IndexAttr) -> StencilBoundsAttr:
        return StencilBoundsAttr(
            zip(
                self.lb + o,
                self.ub + o,
            )
        )

    def __radd__(self, o: IndexAttr) -> StencilBoundsAttr:
        return self + o


@dataclass(frozen=True, init=False)
class StencilType(
    ParametrizedAttribute,
    TypeAttribute,
    builtin.ShapedType,
    builtin.ContainerType[_FieldTypeElement],
    Generic[_FieldTypeElement],
):
    name = "stencil.type"
    bounds: StencilBoundsAttr | IntAttr
    """
    Represents the bounds information of a stencil.field or stencil.temp.

    A StencilBoundsAttr encodes known bounds, where an IntAttr encodes the
    rank of unknown bounds. A stencil.field or stencil.temp cannot be unranked!
    """
    element_type: _FieldTypeElement

    def get_num_dims(self) -> int:
        if isinstance(self.bounds, IntAttr):
            return self.bounds.data
        else:
            return len(self.bounds.ub.array.data)

    def get_shape(self) -> tuple[int, ...]:
        if isinstance(self.bounds, IntAttr):
            return (DYNAMIC_INDEX,) * self.bounds.data
        else:
            return tuple(self.bounds.ub - self.bounds.lb)

    def get_element_type(self) -> _FieldTypeElement:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        def parse_interval() -> tuple[int, int] | int:
            if parser.parse_optional_punctuation("?"):
                return DYNAMIC_INDEX
            parser.parse_punctuation("[")
            l = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation(",")
            u = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation("]")
            return (l, u)

        parser.parse_characters("<")
        bounds = [parse_interval()]
        parser.parse_shape_delimiter()
        opt_type = parser.parse_optional_type()
        while opt_type is None:
            bounds.append(parse_interval())
            parser.parse_shape_delimiter()
            opt_type = parser.parse_optional_type()
        parser.parse_characters(">")
        if isa(bounds, list[tuple[int, int]]):
            bounds = StencilBoundsAttr(bounds)
        elif isa(bounds, list[int]):
            bounds = IntAttr(len(bounds))
        else:
            parser.raise_error("stencil types can only be fully dynamic or sized.")

        return [bounds, opt_type]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if isinstance(self.bounds, StencilBoundsAttr):
                printer.print_list(
                    zip(self.bounds.lb, self.bounds.ub),
                    lambda b: printer.print_string(f"[{b[0]},{b[1]}]"),
                    "x",
                )
                printer.print_string("x")
            else:
                for _ in range(self.bounds.data):
                    printer.print_string("?x")
            printer.print_attribute(self.element_type)

    def __init__(
        self,
        bounds: (
            Iterable[tuple[int | IntAttr, int | IntAttr]]
            | int
            | IntAttr
            | StencilBoundsAttr
        ),
        element_type: _FieldTypeElement,
    ) -> None:
        """
            A StencilBoundsAttr encodes known bounds, where an IntAttr encodes the
        rank of unknown bounds. A stencil.field or stencil.temp cannot be unranked!

        ### examples:

        - `Field(3,f32)` is represented as `stencil.field<?x?x?xf32>`
        - `Field([(-1,17),(-2,18)],f32)` is represented as `stencil.field<[-1,17]x[-2,18]xf32>`,
        """
        if isinstance(bounds, Iterable):
            nbounds = StencilBoundsAttr(bounds)
        elif isinstance(bounds, int):
            nbounds = IntAttr(bounds)
        else:
            nbounds = bounds
        return super().__init__(nbounds, element_type)

    @classmethod
    def constr(
        cls,
        *,
        bounds: AttrConstraint | None = None,
        element_type: AttrConstraint[_FieldTypeElement] | None = None,
    ) -> (
        BaseAttr[StencilType[_FieldTypeElement]]
        | ParamAttrConstraint[StencilType[_FieldTypeElement]]
    ):
        if bounds is None and element_type is None:
            return BaseAttr(cls)
        return ParamAttrConstraint(cls, (bounds, element_type))


@irdl_attr_definition(init=False)
class FieldType(
    StencilType[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    Generic[_FieldTypeElement],
):
    """
    stencil.field represents memory from which stencil input values will be loaded,
    or to which stencil output values will be stored.

    stencil.temp are loaded from or stored to stencil.field
    """

    name = "stencil.field"


@irdl_attr_definition(init=False)
class TempType(
    StencilType[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    Generic[_FieldTypeElement],
):
    """
    stencil.temp represents stencil values, and is the type on which stencil.apply operates.
    It has value-semantics: it won't necesseraly be lowered to an actual buffer.
    """

    name = "stencil.temp"


StencilTypeConstr = FieldType.constr() | TempType.constr()
FieldTypeConstr = FieldType[Attribute].constr()
TempTypeConstr = TempType[Attribute].constr()

AnyTempType: TypeAlias = TempType[Attribute]


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    name = "stencil.result"
    elem: Attribute


class ApplyOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.stencil import (
            ApplyRedundantOperands,
            ApplyUnusedOperands,
            ApplyUnusedResults,
        )

        return (
            ApplyRedundantOperands(),
            ApplyUnusedResults(),
            ApplyUnusedOperands(),
        )


class ApplyOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.shape_inference_patterns.stencil import (
            ApplyOpShapeInference,
        )

        return (ApplyOpShapeInference(),)


class ApplyMemoryEffect(RecursiveMemoryEffect):
    @classmethod
    def get_effects(cls, op: Operation):
        effects = super().get_effects(op)
        if effects is not None:
            for d in cast(ApplyOp, op).dest:
                effects.add(EffectInstance(MemoryEffectKind.WRITE, d))
            for o in cast(ApplyOp, op).args:
                if isinstance(o.type, FieldType):
                    effects.add(EffectInstance(MemoryEffectKind.READ, o))
        return effects


@irdl_op_definition
class ApplyOp(IRDLOperation):
    """
    This operation takes a stencil function plus parameters and applies
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      }

    The computation bounds are defined by the bounds of the output types, which are
    constrained to be all equals.
    """

    name = "stencil.apply"

    args = var_operand_def(Attribute)
    dest = var_operand_def(FieldType)
    region = region_def()
    res = var_result_def(TempType)

    bounds = opt_prop_def(StencilBoundsAttr)

    traits = traits_def(
        IsolatedFromAbove(),
        ApplyOpHasCanonicalizationPatternsTrait(),
        ApplyOpHasShapeInferencePatternsTrait(),
        ApplyMemoryEffect(),
    )

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def print(self, printer: Printer):
        def print_assign_argument(args: tuple[BlockArgument, SSAValue, Attribute]):
            printer.print_ssa_value(args[0])
            printer.print_string(" = ")
            printer.print_ssa_value(args[1])
            printer.print_string(" : ")
            printer.print_attribute(args[2])

        def print_destination_operand(dest: SSAValue):
            printer.print_ssa_value(dest)
            printer.print_string(" : ")
            printer.print_attribute(dest.type)

        with printer.in_parens():
            printer.print_list(
                zip(self.region.block.args, self.args, self.args.types),
                print_assign_argument,
            )
        if self.dest:
            printer.print_string(" outs ")
            with printer.in_parens():
                printer.print_list(self.dest, print_destination_operand)
        else:
            printer.print_string(" -> ")
            with printer.in_parens():
                printer.print_list(self.res.types, printer.print_attribute)
        printer.print_string(" ")
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print_region(self.region, print_entry_block_args=False)
        if self.bounds is not None:
            printer.print_string(" to ")
            self.bounds.print_parameters(printer)

    @classmethod
    def parse(cls: type[ApplyOp], parser: Parser):
        def parse_assign_args():
            arg = parser.parse_argument(expect_type=False)
            parser.parse_punctuation("=")
            value = parser.parse_operand()
            parser.parse_punctuation(":")
            type = parser.parse_attribute()
            arg = arg.resolve(type)
            return arg, value

        def parse_operand():
            op = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            type = parser.parse_attribute()
            return parser.resolve_operand(op, type)

        assign_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parse_assign_args
        )
        args: tuple[Parser.Argument, ...]
        operands: tuple[SSAValue, ...]
        args, operands = zip(*assign_args) if assign_args else ((), ())

        if parser.parse_optional_punctuation("->"):
            parser.parse_punctuation("(")
            result_types = parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_attribute, parser.parse_attribute
            )
            destinations = []
        else:
            parser.parse_keyword("outs")
            parser.parse_punctuation("(")
            destinations = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parse_operand
            )
            result_types = []
        parser.parse_punctuation(")")
        attrs = parser.parse_optional_attr_dict_with_keyword()
        if attrs is not None:
            attrs = dict(attrs.data)
        else:
            attrs = {}
        region = parser.parse_region(args)
        if parser.parse_optional_keyword("to"):
            bounds = StencilBoundsAttr.new(StencilBoundsAttr.parse_parameters(parser))
        else:
            bounds = None
        return cls(
            operands=[operands, destinations or []],
            result_types=[result_types or []],
            regions=[region],
            attributes=attrs,
            properties={"bounds": bounds},
        )

    @staticmethod
    def get(
        args: Sequence[SSAValue] | Sequence[Operation],
        body: Block | Region,
        result_types: Sequence[TempType[Attribute]] = (),
        bounds: StencilBoundsAttr | None = None,
    ):
        assert result_types or bounds
        if isinstance(body, Block):
            body = Region(body)

        properties = {"bounds": bounds} if bounds else {}

        return ApplyOp.build(
            operands=[list(args), []],
            regions=[body],
            result_types=[result_types],
            properties=properties,
        )

    def verify_(self) -> None:
        for operand, argument in zip(self.operands, self.region.block.args):
            if operand.type != argument.type:
                raise VerifyException(
                    "Expected argument type to match operand type, got "
                    f"{argument.type} != {operand.type} at index {argument.index}"
                )
        if len(self.res) > 0 and len(self.dest) > 0:
            raise VerifyException(
                "Expected stencil.apply to have all value-semantics result or "
                "buffer-semantic destination operands."
            )
        if len(self.res) > 0:
            res_type = self.res[0].type
            for other in self.res[1:]:
                other = other.type
                if res_type.bounds != other.bounds:
                    raise VerifyException(
                        "Expected all output types bounds to be equals."
                    )
        if len(self.dest) > 0:
            if self.bounds is None:
                raise VerifyException(
                    "Expected stencil.apply to have bounds when having destination operands."
                )

        nres = max(len(self.res), len(self.dest))
        if nres < 1:
            raise VerifyException(
                f"Expected stencil.apply to have at least 1 result, got {len(self.res)}"
            )

    def get_rank(self) -> int:
        if len(self.res) > 0:
            res_type = self.res[0].type
            assert isa(res_type, TempType[Attribute])
            return res_type.get_num_dims()
        else:
            assert self.bounds is not None
            return len(self.bounds.lb)

    def get_accesses(self) -> Iterable[AccessPattern]:
        """
        Return the access patterns of each input.

         - An offset is a tuple describing a relative access
         - An access pattern is a class wrapping a sequence of offsets
         - This method returns an access pattern for each stencil
           field of the apply operation.
        """
        # iterate over the block arguments
        for arg in self.region.block.args:
            accesses: list[tuple[int, ...]] = []
            # walk the uses of the argument
            for use in arg.uses:
                # filter out all non access ops
                if not isinstance(use.operation, AccessOp):
                    continue
                access: AccessOp = use.operation
                # grab the offsets as a tuple[int, ...]
                offsets = tuple(access.offset)
                # account for offset_mappings:
                if (
                    access.offset_mapping is not None
                    and len(offsets) == self.get_rank()
                ):
                    offsets = tuple(offsets[i] for i in access.offset_mapping)
                accesses.append(offsets)
            yield AccessPattern(tuple(accesses))

    def get_bounds(self):
        if self.bounds is not None:
            return self.bounds
        else:
            assert self.res
            res_type = self.res[0].type
            return res_type.bounds


class AllocOpEffect(MemoryEffect):
    @classmethod
    def get_effects(cls, op: Operation):
        return {EffectInstance(MemoryEffectKind.ALLOC, cast(AllocOp, op).field)}


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "stencil.alloc"

    field = result_def(FieldType[Attribute])

    assembly_format = "attr-dict `:` type($field)"

    traits = traits_def(AllocOpEffect())


class CastOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.stencil import (
            RemoveCastWithNoEffect,
        )

        return (RemoveCastWithNoEffect(),)


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
        %0 = stencil.cast %in : !stencil.field<?x?x?xf64> -> !stencil.field<70x70x60xf64> # noqa
    """

    name = "stencil.cast"

    field = operand_def(
        FieldType[Attribute].constr(
            element_type=MessageConstraint(
                VarConstraint("T", AnyAttr()),
                "Input and output fields must have the same element types",
            )
        )
    )
    result = result_def(
        FieldType[Attribute].constr(
            element_type=MessageConstraint(
                VarConstraint("T", AnyAttr()),
                "Input and output fields must have the same element types",
            )
        )
    )

    assembly_format = (
        "$field attr-dict-with-keyword `:` type($field) `->` type($result)"
    )

    traits = traits_def(NoMemoryEffect(), CastOpHasCanonicalizationPatternsTrait())

    @staticmethod
    def get(
        field: SSAValue | Operation,
        bounds: StencilBoundsAttr,
        res_type: FieldType[_FieldTypeElement] | FieldType[Attribute] | None = None,
    ) -> CastOp:
        """ """
        field_ssa = SSAValue.get(field, type=FieldType)
        if res_type is None:
            res_type = FieldType(
                bounds,
                field_ssa.type.element_type,
            )
        return CastOp.build(
            operands=[field],
            result_types=[res_type],
        )

    def verify_(self) -> None:
        # this should be fine, verify() already checks them:
        assert isa(self.field.type, FieldType[Attribute])
        assert isa(self.result.type, FieldType[Attribute])
        if self.field.type.get_num_dims() != self.result.type.get_num_dims():
            raise VerifyException("Input and output types must have the same rank")

        if (
            isinstance(self.field.type.bounds, StencilBoundsAttr)
            and self.field.type.bounds != self.result.type.bounds
        ):
            raise VerifyException(
                "If input shape is not dynamic, it must be the same as output"
            )


class CombineOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.shape_inference_patterns.stencil import (
            CombineOpShapeInference,
        )

        return (CombineOpShapeInference(),)


@irdl_op_definition
class CombineOp(IRDLOperation):
    """
        Combines the results computed on a lower with the results computed on
        an upper domain. The operation combines the domain at a given index/offset
        in a given dimension. Optional extra operands allow to combine values
        that are only written / defined on the lower or upper subdomain. The result
        values have the order upper/lower, lowerext, upperext.

        Example:
          `%res1, %res2 = stencil.combine 1 at 11 lower = (%0 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>) lowerext = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>`
        Can be illustrated as:
    ```
        dim   1       offset                   offset
             ┌──►      (=11)                    (=11)
           0 │          │                        │
             ▼ ┌────────┼─────────┐     ┌────────┼─────────┐
               │        │         │     │        │         │
               │        │         │     │        │         │
          %res1│  lower │ upper   │     │lowerext│         │%res2
               │    %0  │   %1    │     │    %0  │         │
               │        │         │     │        │         │
               │        │         │     │        │         │
               └────────┼─────────┘     └────────┼─────────┘
                        │                        │
    ```
    """  # noqa: E501

    name = "stencil.combine"

    dim = attr_def(IntegerAttr[IndexType])
    index = attr_def(IntegerAttr[IndexType])
    lower = var_operand_def(TempType)
    upper = var_operand_def(TempType)
    lowerext = var_operand_def(TempType)
    upperext = var_operand_def(TempType)
    results_ = var_result_def(TempType)

    traits = traits_def(
        Pure(),
        CombineOpHasShapeInferencePatternsTrait(),
    )

    assembly_format = "$dim `at` $index `lower` `=` `(` $lower `:` type($lower) `)` `upper` `=` `(` $upper `:` type($upper) `)` (`lowerext` `=` $lowerext^ `:` type($lowerext))? (`upperext` `=` $upperext^ `:` type($upperext))? attr-dict-with-keyword `:` type($results_)"  # noqa: E501

    irdl_options = [AttrSizedOperandSegments()]


class DynAccessOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls):
        from xdsl.transforms.shape_inference_patterns.stencil import (
            DynAccessOpShapeInference,
        )

        return (DynAccessOpShapeInference(),)


@irdl_op_definition
class DynAccessOp(IRDLOperation):
    """
    This operation accesses a temporary element given a dynamic offset.
    The offset is specified in absolute coordinates. An additional
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation.

    Example:
      %0 = stencil.dyn_access %temp [%i, %j, %k] in [-1, -1, -1] : [1, 1, 1] : !stencil.temp<?x?x?xf64>
    """

    name = "stencil.dyn_access"

    temp = operand_def(
        StencilType[Attribute].constr(
            element_type=MessageConstraint(
                VarConstraint("T", AnyAttr()),
                "Expected result type to be the accessed temp's element type.",
            )
        )
    )

    offset = var_operand_def(builtin.IndexType())
    lb = attr_def(IndexAttr)
    ub = attr_def(IndexAttr)

    res = result_def(
        MessageConstraint(
            VarConstraint("T", AnyAttr()),
            "Expected result type to be the accessed temp's element type.",
        )
    )

    assembly_format = (
        "$temp `[` $offset `]` `in` $lb `:` $ub attr-dict-with-keyword `:` type($temp)"
    )

    traits = traits_def(
        HasAncestor(ApplyOp),
        NoMemoryEffect(),
        DynAccessOpHasShapeInferencePatternsTrait(),
    )

    def __init__(
        self,
        temp: SSAValue | Operation,
        offset: Sequence[SSAValue | Operation],
        lb: IndexAttr,
        ub: IndexAttr,
    ):
        temp_type = SSAValue.get(temp, type=TempType).type
        super().__init__(
            operands=[temp, list(offset)],
            attributes={"lb": lb, "ub": ub},
            result_types=[temp_type.element_type],
        )


@irdl_op_definition
class ExternalLoadOp(IRDLOperation):
    """
    This operation loads from an external field type, e.g. to bring data into the stencil

    Example:
      %0 = stencil.external_load %in : !fir.array<128x128xf64> -> !stencil.field<128x128xf64>
    """

    name = "stencil.external_load"
    field = operand_def(Attribute)
    result = result_def(base(FieldType[Attribute]) | MemRefType.constr())

    assembly_format = (
        "$field attr-dict-with-keyword `:` type($field) `->` type($result)"
    )

    @staticmethod
    def get(
        arg: SSAValue | Operation,
        res_type: FieldType[Attribute] | memref.MemRefType,
    ):
        return ExternalLoadOp.build(operands=[arg], result_types=[res_type])


@irdl_op_definition
class ExternalStoreOp(IRDLOperation):
    """
    This operation takes a stencil field and then stores this to an external type

    Example:
      stencil.store %temp to %field : !stencil.field<128x128xf64> to !fir.array<128x128xf64> # noqa
    """

    name = "stencil.external_store"
    temp = operand_def(FieldType)
    field = operand_def(Attribute)

    assembly_format = (
        "$temp `to` $field attr-dict-with-keyword `:` type($temp) `to` type($field)"
    )


@irdl_op_definition
class IndexOp(IRDLOperation):
    """
    This operation returns the index of the current loop iteration for the
    chosen direction (0, 1, or 2).
    The offset is specified relative to the current position.

    Example:
      %0 = stencil.index 0 [-1, 0, 0]
    """

    name = "stencil.index"
    dim = attr_def(IntegerAttr[IndexType])
    offset = attr_def(IndexAttr)
    idx = result_def(builtin.IndexType())

    assembly_format = "$dim $offset attr-dict-with-keyword"

    traits = traits_def(HasAncestor(ApplyOp), Pure())

    def get_apply(self):
        """
        Simple helper to get the parent apply and raise otherwise.
        """
        trait = self.get_trait(HasAncestor(ApplyOp))
        assert trait is not None
        ancestor = trait.get_ancestor(self)
        if ancestor is None:
            raise ValueError(
                "stencil.apply not found, this function should be called on"
                "verified accesses only."
            )
        return cast(ApplyOp, ancestor)


class AccessOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.shape_inference_patterns.stencil import (
            AccessOpShapeInference,
        )

        return (AccessOpShapeInference(),)


@irdl_op_definition
class AccessOp(IRDLOperation):
    """
    This operation accesses a value from a stencil.temp given the specified offset.
    offset. The offset is specified relative to the current position.

    The optional offset mapping will determine which offset corresponds to which
    result dimension and is needed when we are accessing an array which has fewer
    dimensions than the result.

    Example:
      %0 = stencil.access %temp[-1, 0, 0] : !stencil.temp<?x?x?xf64>
    """

    name = "stencil.access"
    temp = operand_def(
        StencilType[Attribute].constr(
            element_type=MessageConstraint(
                VarConstraint("T", AnyAttr()),
                "Expected return type to match the accessed temp's element type.",
            )
        )
    )
    offset = attr_def(IndexAttr)
    offset_mapping = opt_attr_def(IndexAttr)
    res = result_def(
        MessageConstraint(
            VarConstraint("T", AnyAttr()),
            "Expected return type to match the accessed temp's element type.",
        )
    )

    traits = traits_def(
        HasAncestor(ApplyOp), Pure(), AccessOpHasShapeInferencePatternsTrait()
    )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.temp)
        printer.print_op_attributes(
            self.attributes,
            reserved_attr_names={"offset", "offset_mapping"},
            print_keyword=True,
        )

        # IRDL-enforced, not supposed to use custom syntax if not verified
        trait = AccessOp.get_trait(HasAncestor(ApplyOp))
        assert trait is not None
        apply = cast(ApplyOp, trait.get_ancestor(self))

        mapping = self.offset_mapping
        if mapping is None:
            mapping = range(apply.get_rank())
        offset = list(self.offset)

        with printer.in_square_brackets():
            index = 0
            for i in range(apply.get_rank()):
                if i in mapping:
                    printer.print_int(offset[index])
                    index += 1
                else:
                    printer.print_string("_")
                if i != apply.get_rank() - 1:
                    printer.print_string(", ")

        printer.print_string(" : ")
        printer.print_attribute(self.temp.type)

    @classmethod
    def parse(cls, parser: Parser):
        temp = parser.parse_operand()

        index = 0
        offset = list[int]()
        offset_mapping = list[int]()
        parser.parse_punctuation("[")
        while True:
            o = parser.parse_optional_integer()
            if o is None:
                parser.parse_characters("_")
            else:
                offset.append(o)
                offset_mapping.append(index)
            if parser.parse_optional_punctuation("]"):
                break
            parser.parse_punctuation(",")
            index += 1

        attrs = parser.parse_optional_attr_dict_with_keyword(
            {"offset", "offset_mapping"}
        )
        attrs = dict(attrs.data) if attrs else {}
        attrs["offset"] = IndexAttr.get(*offset)
        if offset_mapping:
            attrs["offset_mapping"] = IndexAttr.get(*offset_mapping)
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        if not isa(res_type, StencilType):
            parser.raise_error(
                "Expected return type to be a stencil.temp or stencil.field"
            )
        return cls.build(
            operands=[temp], result_types=[res_type.element_type], attributes=attrs
        )

    @staticmethod
    def get(
        temp: SSAValue | Operation,
        offset: Sequence[int],
        offset_mapping: Sequence[int] | IndexAttr | None = None,
    ):
        temp_type = SSAValue.get(temp, type=StencilType).type

        attributes: dict[str, Attribute] = {
            "offset": IndexAttr(
                ArrayAttr(IntAttr(value) for value in offset),
            ),
        }

        if offset_mapping is not None:
            attributes["offset_mapping"] = IndexAttr.get(*offset_mapping)

        return AccessOp.build(
            operands=[temp],
            attributes=attributes,
            result_types=[temp_type.element_type],
        )

    def verify_(self) -> None:
        # As promised by HasAncestor(ApplyOp)
        trait = AccessOp.get_trait(HasAncestor(ApplyOp))
        assert trait is not None
        apply = trait.get_ancestor(self)
        assert isinstance(apply, ApplyOp)

        # TODO This should be handled by infra, having a way to verify things on ApplyOp
        # **before** its children.
        # cf https://github.com/xdslproject/xdsl/issues/1112
        apply.verify_()

        temp_type = self.temp.type
        assert isa(temp_type, StencilType)
        if temp_type.get_num_dims() != apply.get_rank():
            if self.offset_mapping is None:
                raise VerifyException(
                    f"Expected stencil.access operand to be of rank {apply.get_rank()} "
                    f"to match its parent apply, got {temp_type.get_num_dims()} without "
                    f"explict offset mapping provided"
                )

        if self.offset_mapping is not None and len(self.offset_mapping) != len(
            self.offset
        ):
            raise VerifyException(
                f"Expected stencil.access offset mapping be of length {len(self.offset)} "
                f"to match the provided offsets, but it is {len(self.offset_mapping)} "
                f"instead"
            )

        if self.offset_mapping is not None:
            prev_offset = None
            for prev_offset, offset in pairwise(self.offset_mapping):
                if prev_offset >= offset:
                    raise VerifyException(
                        "Offset mapping in stencil.access must be strictly increasing."
                        "increasing"
                    )
            for offset in self.offset_mapping:
                if offset >= apply.get_rank():
                    raise VerifyException(
                        f"Offset mappings in stencil.access must be within the rank of the "
                        f"apply, got {offset} >= {apply.get_rank()}"
                    )

        if len(self.offset) != temp_type.get_num_dims():
            raise VerifyException(
                f"Expected offset's rank to be {temp_type.get_num_dims()} to match the "
                f"operand's rank, got {len(self.offset)}"
            )

    def get_apply(self):
        """
        Simple helper to get the parent apply and raise otherwise.
        """
        trait = self.get_trait(HasAncestor(ApplyOp))
        assert trait is not None
        ancestor = trait.get_ancestor(self)
        if ancestor is None:
            raise ValueError(
                "stencil.apply not found, this function should be called on"
                "verified accesses only."
            )
        return cast(ApplyOp, ancestor)


class LoadOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.shape_inference_patterns.stencil import (
            LoadOpShapeInference,
        )

        return (LoadOpShapeInference(),)


class LoadOpMemoryEffect(MemoryEffect):
    @classmethod
    def get_effects(cls, op: Operation):
        return {EffectInstance(MemoryEffectKind.READ, cast(LoadOp, op).field)}


class TensorIgnoreSizeConstraint(VarConstraint[Attribute]):
    @staticmethod
    def ranks_and_element_types_match(attr: TensorType, other: Attribute) -> bool:
        return (
            isa(other, TensorType)
            and len(attr.get_shape()) == len(other.get_shape())
            and attr.get_element_type() == other.get_element_type()
        )

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        ctx_attr = constraint_context.get_variable(self.name)
        if ctx_attr is not None:
            if isa(
                attr, TensorType[Attribute]
            ) and TensorIgnoreSizeConstraint.ranks_and_element_types_match(
                attr, ctx_attr
            ):
                return
        super().verify(attr, constraint_context)


@irdl_op_definition
class LoadOp(IRDLOperation):
    """
    This operation takes a field and returns its values.

    Example:
      %0 = stencil.load %field : !stencil.field<70x70x60xf64> -> !stencil.temp<?x?x?xf64>
    """

    name = "stencil.load"

    field = operand_def(
        FieldType[Attribute].constr(
            bounds=base(StencilBoundsAttr),
            element_type=MessageConstraint(
                TensorIgnoreSizeConstraint("T", AnyAttr()),
                "Expected element types to match.",
            ),
        )
    )
    res = result_def(
        TempType[Attribute].constr(
            element_type=MessageConstraint(
                TensorIgnoreSizeConstraint("T", AnyAttr()),
                "Expected element types to match.",
            )
        )
    )

    assembly_format = "$field attr-dict-with-keyword `:` type($field) `->` type($res)"

    traits = traits_def(LoadOpHasShapeInferencePatternsTrait(), LoadOpMemoryEffect())

    @staticmethod
    def get(
        field: SSAValue | Operation,
        lb: IndexAttr | None = None,
        ub: IndexAttr | None = None,
    ):
        field_type = SSAValue.get(field, type=FieldType).type

        if lb is None or ub is None:
            res_type = TempType(field_type.get_num_dims(), field_type.element_type)
        else:
            res_type = TempType(zip(lb, ub), field_type.element_type)

        return LoadOp.build(
            operands=[field],
            result_types=[res_type],
        )

    def verify_(self) -> None:
        field = self.field.type
        temp = self.res.type
        assert isa(field, FieldType[Attribute])
        assert isa(temp, TempType[Attribute])
        if isinstance(field.bounds, StencilBoundsAttr) and isinstance(
            temp.bounds, StencilBoundsAttr
        ):
            if temp.bounds.lb < field.bounds.lb or temp.bounds.ub > field.bounds.ub:
                raise VerifyException(
                    "The stencil.load is too big for the loaded field."
                )


class BufferOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.shape_inference_patterns.stencil import (
            BufferOpShapeInference,
        )

        return (BufferOpShapeInference(),)


@irdl_op_definition
class BufferOp(IRDLOperation):
    """
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>)
    """

    name = "stencil.buffer"

    temp = operand_def(
        TempType[Attribute].constr(
            bounds=MessageConstraint(
                VarConstraint("B", AnyAttr()),
                "Expected input and output to have the same bounds",
            ),
            element_type=MessageConstraint(
                VarConstraint("E", AnyAttr()),
                "Expected input and output to have the same element type",
            ),
        )
    )
    res = result_def(
        StencilType[Attribute].constr(
            bounds=MessageConstraint(
                VarConstraint("B", AnyAttr()),
                "Expected input and output to have the same bounds",
            ),
            element_type=MessageConstraint(
                VarConstraint("E", AnyAttr()),
                "Expected input and output to have the same element type",
            ),
        )
    )

    assembly_format = "$temp attr-dict-with-keyword `:` type($temp) `->` type($res)"

    traits = traits_def(Pure(), BufferOpHasShapeInferencePatternsTrait())

    def __init__(self, temp: SSAValue | Operation):
        temp = SSAValue.get(temp)
        super().__init__(operands=[temp], result_types=[temp.type])

    def verify_(self) -> None:
        # When used as a bufferization op, it should be flexible.
        # This is probably something you don't want to see, but should be valid - it just
        # means bufferization was incomplete.
        if isinstance(self.res.type, FieldType):
            return
        if not isinstance(self.temp.owner, ApplyOp | CombineOp):
            owner = (
                "block argument"
                if isinstance(self.temp.owner, Block)
                else self.temp.owner.name
            )
            raise VerifyException(
                f"Expected stencil.buffer operand to be a result of stencil.apply or stencil.combine got {owner}"
            )
        if any(not isinstance(use.operation, BufferOp) for use in self.temp.uses):
            raise VerifyException(
                "A stencil.buffer's operand temp should only be buffered. You can use "
                "stencil.buffer's output instead!"
            )


class StoreOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.shape_inference_patterns.stencil import (
            StoreOpShapeInference,
        )

        return (StoreOpShapeInference(),)


class StoreOpMemoryEffect(MemoryEffect):
    @classmethod
    def get_effects(cls, op: Operation):
        return {EffectInstance(MemoryEffectKind.WRITE, cast(StoreOp, op).field)}


@irdl_op_definition
class StoreOp(IRDLOperation):
    """
    This operation writes values to a field on a user defined range.

    Example:
      stencil.store %temp to %field ([-3, -3, 0] : [67, 67, 60]): !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>  # noqa
    """

    name = "stencil.store"

    temp = operand_def(
        TempType[Attribute].constr(
            element_type=MessageConstraint(
                TensorIgnoreSizeConstraint("T", AnyAttr()),
                "Input and output fields must have the same element types",
            ),
        )
    )
    field = operand_def(
        FieldType[Attribute].constr(
            bounds=MessageConstraint(
                StencilBoundsAttr, "Output type's size must be explicit"
            ),
            element_type=MessageConstraint(
                TensorIgnoreSizeConstraint("T", AnyAttr()),
                "Input and output fields must have the same element types",
            ),
        )
    )
    bounds = attr_def(StencilBoundsAttr)

    assembly_format = "$temp `to` $field `` `(` $bounds `)` attr-dict-with-keyword `:` type($temp) `to` type($field)"

    traits = traits_def(StoreOpHasShapeInferencePatternsTrait(), StoreOpMemoryEffect())

    @staticmethod
    def get(
        temp: SSAValue | Operation,
        field: SSAValue | Operation,
        bounds: StencilBoundsAttr,
    ):
        return StoreOp.build(operands=[temp, field], attributes={"bounds": bounds})


@irdl_op_definition
class StoreResultOp(IRDLOperation):
    """
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>
    """

    name = "stencil.store_result"

    arg = opt_operand_def(
        MessageConstraint(
            VarConstraint("T", AnyAttr()),
            "Expected return type to carry the operand type.",
        )
    )
    res = result_def(
        ParamAttrConstraint(
            ResultType,
            [
                MessageConstraint(
                    VarConstraint("T", AnyAttr()),
                    "Expected return type to carry the operand type.",
                )
            ],
        )
    )

    assembly_format = "$arg attr-dict-with-keyword `:` type($res)"

    traits = traits_def(HasAncestor(ApplyOp), Pure())


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    The return operation terminates the stencil.apply and writes
    the results of the stencil operator to the temporary values returned
    by the stencil.apply operation. The types and the number of operands
    must match the results of the stencil.apply operation.

    The optional unroll attribute enables the implementation of loop
    unrolling at the stencil dialect level.

    Examples:
      stencil.return %0 : !stencil.result<f64>
    """

    name = "stencil.return"

    arg = var_operand_def(Attribute)
    unroll = opt_prop_def(IndexAttr)

    assembly_format = "$arg (`unroll` $unroll^)? attr-dict-with-keyword `:` type($arg)"

    @property
    def unroll_factor(self) -> int:
        if self.unroll is None:
            return 1
        return prod(self.unroll)

    traits = traits_def(HasParent(ApplyOp), IsTerminator(), Pure())

    @staticmethod
    def get(res: Sequence[SSAValue | Operation]):
        return ReturnOp.build(operands=[list(res)])

    def verify_(self) -> None:
        unroll_factor = self.unroll_factor
        types = [ot.elem if isinstance(ot, ResultType) else ot for ot in self.arg.types]
        apply = cast(ApplyOp, self.parent_op())
        if len(apply.res) > 0:
            res_types = [r.type.element_type for r in apply.res]
        else:
            res_types = [
                cast(FieldType[Attribute], o.type).element_type for o in apply.dest
            ]
        if len(types) != len(res_types) * unroll_factor:
            raise VerifyException(
                f"stencil.return expected {len(res_types) * unroll_factor} operands to match the parent "
                f"stencil.apply result types, got {len(types)}"
            )
        # stencil.return returns `unroll_factor` values per stencil.apply result
        # This checks types are consistent for each of those.
        for i, res_type in enumerate(res_types):
            for j in range(unroll_factor * i, unroll_factor * (i + 1)):
                op_type = types[j]
                if op_type != res_type and not (
                    isa(op_type, TensorType)
                    and TensorIgnoreSizeConstraint.ranks_and_element_types_match(
                        op_type, res_type
                    )
                ):
                    raise VerifyException(
                        "stencil.return expected operand types to match the parent "
                        f"stencil.apply result element types. Got {op_type} at index "
                        f"{j}, expected {res_type}."
                    )


@dataclass(frozen=True)
class AccessPattern:
    """
    Represents access patterns of a stencil.apply operation.

    Contains helpers to get common information about accesses such as diagonals.
    """

    offsets: tuple[tuple[int, ...], ...]

    @property
    def dims(self):
        """
        Dimensionality of the accesses.
        """
        if not self.offsets:
            return 0
        return len(self.offsets[0])

    @property
    def is_diagonal(self) -> bool:
        """
        Check if the access pattern has diagonal accesses.
        """
        for _ in self.get_diagonals():
            return True
        return False

    def get_diagonals(self, degree: int = 2) -> Iterable[tuple[int, ...]]:
        """
        Returns all offsets that have <degree=2> or more non-zero entries.

        For <degree> >= 2 this makes them diagonals.

        For <degree> = 1 it returns all accesses that are nonzero.
        """
        for ax in self.offsets:
            # get the number of nonzero entries in offset
            if sum(1 if x != 0 else 0 for x in ax) >= degree:
                yield ax

    def halo_in_axis(self, axis: int) -> tuple[int, int]:
        """
        Returns the minimum and maximum access distance for a single axis.
        """
        left, right = 0, 0
        for ax in self.offsets:
            left = min(ax[axis], left)
            right = max(ax[axis], right)
        return left, right

    def halos(self) -> tuple[tuple[int, int], ...]:
        """
        Return a tuple containing the maximum and minimum offsets in each axis.
        E.g. ((-2, 2), (-1, 1)) represents a pattern that accesses cells at most
        (-2, 2) away in the x-axis, and -1, 1 away in the y-axis.
        """
        n = self.dims
        lefts, rights = [0] * n, [0] * n
        for ax in self.offsets:
            for axis in range(n):
                lefts[axis] = min(ax[axis], lefts[axis])
                rights[axis] = max(ax[axis], rights[axis])
        return tuple(zip(lefts, rights))

    def max_distance(self) -> int:
        """
        Returns the maximum absolute accessed distance across all axes.
        """
        res = 0
        for ax in self.offsets:
            res = max(res, max(abs(a) for a in ax))
        return res

    def visual_pattern(self) -> str:
        """
        Returns a visual equivalent of the access pattern, only works for 1d and 2d.

        Returns patterns where O signifies the center point and X represents an access.
        E.g.:

         X
        XOX
         X

        For a 2d-4pt stencil.
        """
        # handle special cases:
        if self.dims == 0:
            return "O"
        elif self.dims > 2:
            return "Too many dimensions in access"
        elif self.dims == 1:
            halos = (self.halo_in_axis(0), (0, 0))
        else:
            halos = self.halos()
        x_axis_halo, y_axis_halo = halos
        # construct a matrix of the required size:
        points = [
            [" " for _ in range(y_axis_halo[1] - y_axis_halo[0] + 1)]
            for __ in range(x_axis_halo[1] - x_axis_halo[0] + 1)
        ]
        # set the center point:
        points[-x_axis_halo[0]][-y_axis_halo[0]] = "O"
        # set each access:
        for access in self.offsets:
            points[access[0] - x_axis_halo[0]][access[1] - y_axis_halo[0]] = "X"
        # construct the string:
        return "\n".join("".join(row) for row in points)


Stencil = Dialect(
    "stencil",
    [
        AllocOp,
        CastOp,
        CombineOp,
        DynAccessOp,
        ExternalLoadOp,
        ExternalStoreOp,
        IndexOp,
        AccessOp,
        LoadOp,
        BufferOp,
        StoreOp,
        ApplyOp,
        StoreResultOp,
        ReturnOp,
    ],
    [
        FieldType,
        TempType,
        ResultType,
        IndexAttr,
        StencilBoundsAttr,
    ],
)
