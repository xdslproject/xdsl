from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import ClassVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AffineSetAttr,
    ArrayAttr,
    ContainerType,
    DenseIntElementsAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ShapedType,
    StringAttr,
    VectorType,
)
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.utils import print_assignment
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
    AffineSymExpr,
)
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    IsTerminator,
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_op_definition
class ApplyOp(IRDLOperation):
    name = "affine.apply"

    mapOperands = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)
    result = result_def(IndexType)

    traits = traits_def(Pure())

    def __init__(self, map_operands: Sequence[SSAValue], affine_map: AffineMapAttr):
        super().__init__(
            operands=[map_operands],
            properties={"map": affine_map},
            result_types=[IndexType()],
        )

    def verify_(self) -> None:
        if len(self.mapOperands) != self.map.data.num_dims + self.map.data.num_symbols:
            raise VerifyException(
                f"{self.name} expects "
                f"{self.map.data.num_dims + self.map.data.num_symbols} operands, but "
                f"got {len(self.mapOperands)}. The number of map operands must match "
                "the sum of the dimensions and symbols of its map."
            )
        if len(self.map.data.results) != 1:
            raise VerifyException("affine.apply expects a unidimensional map.")

    @classmethod
    def parse(cls, parser: Parser) -> ApplyOp:
        pos = parser.pos
        m = parser.parse_attribute()
        if not isinstance(m, AffineMapAttr):
            parser.raise_error("Expected affine map attr", at_position=pos)
        dims = parser.parse_optional_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parser.parse_operand()
        )
        if dims is None:
            dims = []
        syms = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: parser.parse_operand()
        )
        if syms is None:
            syms = []
        return ApplyOp(dims + syms, m)

    def print(self, printer: Printer):
        m = self.map.data
        operands = tuple(self.mapOperands)
        assert len(operands) == m.num_dims + m.num_symbols, f"{len(operands)} {m}"
        printer.print_string(" ")
        printer.print_attribute(self.map)
        printer.print_string(" (")
        if m.num_dims:
            printer.print_list(
                operands[: m.num_dims], lambda el: printer.print_operand(el)
            )
        printer.print_string(")")

        if m.num_symbols:
            printer.print_string("[")
            printer.print_list(
                operands[m.num_dims :], lambda el: printer.print_operand(el)
            )
            printer.print_string("]")


def _parse_affine_for_bound(
    parser: Parser, keyword: str
) -> tuple[AffineMapAttr, Sequence[SSAValue]]:
    """
    Parses one `affine.for` loop bound:
    ```
    affine-for-bound ::= `max`|`min`)? affine-map `(` ssa-use-list `)`
                          (`[` ssa-use-list `]`)?
                        | ssa-id
                        | integer-literal
    ```
    The `max`/`min` keyword is mandatory only when the map has multiple results,
    since it otherwise has no effect on a single-result map.

    Mirrors MLIR's [`parseBound`](https://github.com/llvm/llvm-project/blob/99f7018958ed3daf2abf8d49178c24fbf1eb1010/mlir/lib/Dialect/Affine/IR/AffineOps.cpp#L2255).
    """
    used_min_max = parser.parse_optional_keyword(keyword) is not None

    if (operand := parser.parse_optional_operand()) is not None:
        # A bare SSA value is sugar for a single-symbol identity map.
        return (
            AffineMapAttr(AffineMap(0, 1, (AffineExpr.symbol(0),))),
            (operand,),
        )

    pos = parser.pos
    if (value := parser.parse_optional_integer()) is not None:
        return AffineMapAttr(AffineMap.constant_map(value)), ()

    map_attr = parser.parse_attribute()
    if not isinstance(map_attr, AffineMapAttr):
        parser.raise_error("expected an affine map or an integer for loop bound", pos)
    m = map_attr.data

    dims = parser.parse_comma_separated_list(
        Parser.Delimiter.PAREN, parser.parse_operand
    )
    syms = parser.parse_optional_comma_separated_list(
        Parser.Delimiter.SQUARE, parser.parse_operand
    )
    if syms is None:
        syms = []

    if len(dims) != m.num_dims:
        parser.raise_error("dim operand count and affine map dim count must match", pos)
    if len(syms) != m.num_symbols:
        parser.raise_error(
            "symbol operand count and affine map symbol count must match", pos
        )
    if len(m.results) > 1 and not used_min_max:
        parser.raise_error(
            f"loop bound affine map with multiple results requires '{keyword}' prefix",
            pos,
        )

    return map_attr, (*dims, *syms)


def _print_affine_for_bound(
    printer: Printer,
    bound_map: AffineMapAttr,
    bound_operands: Sequence[SSAValue],
    prefix: str,
) -> None:
    """
    Prints one `affine.for` loop bound. Bounds that are a single constant,
    or a single-symbol identity map, are printed in their short forms,
    everything else falls back to the full affine map + operand list,
    with a `max`/`min` prefix when the map has multiple results.

    Mirror's MLIR's [`printBound`](https://github.com/llvm/llvm-project/blob/99f7018958ed3daf2abf8d49178c24fbf1eb1010/mlir/lib/Dialect/Affine/IR/AffineOps.cpp#L2430).
    """
    m = bound_map.data

    if len(m.results) == 1:
        expr = m.results[0]

        no_dims = m.num_dims == 0
        no_syms = m.num_symbols == 0

        # just a constant
        if no_dims and no_syms and isinstance(expr, AffineConstantExpr):
            printer.print_string(str(expr.value))
            return

        # just a symbol
        if no_dims and m.num_symbols == 1 and isinstance(expr, AffineSymExpr):
            printer.print_ssa_value(bound_operands[0])
            return

    else:
        printer.print_string(f"{prefix} ")

    printer.print_attribute(bound_map)

    with printer.in_parens():
        printer.print_list(bound_operands[: m.num_dims], printer.print_ssa_value)

    if m.num_symbols:
        with printer.in_square_brackets():
            printer.print_list(bound_operands[m.num_dims :], printer.print_ssa_value)


@irdl_op_definition
class ForOp(IRDLOperation):
    name = "affine.for"

    lowerBoundOperands = var_operand_def(IndexType)
    upperBoundOperands = var_operand_def(IndexType)
    inits = var_operand_def()
    res = var_result_def()

    lowerBoundMap = prop_def(AffineMapAttr)
    upperBoundMap = prop_def(AffineMapAttr)
    step = prop_def(IntegerAttr)

    body = region_def()

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    # TODO this requires the ImplicitAffineTerminator trait instead of
    # NoTerminator
    # gh issue: https://github.com/xdslproject/xdsl/issues/1149

    def verify_(self) -> None:
        if len(self.inits) != len(self.results):
            raise VerifyException("Expected as many init operands as results.")
        if len(self.lowerBoundOperands) != (
            self.lowerBoundMap.data.num_dims + self.lowerBoundMap.data.num_symbols
        ):
            raise VerifyException(
                "Expected as many lower bound operands as lower bound dimensions and symbols."
            )
        if len(self.upperBoundOperands) != (
            self.upperBoundMap.data.num_dims + self.upperBoundMap.data.num_symbols
        ):
            raise VerifyException(
                "Expected as many upper bound operands as upper bound dimensions and symbols."
            )
        iter_types = self.inits.types
        if iter_types != self.result_types:
            raise VerifyException(
                "Expected all operands and result pairs to have matching types"
            )
        entry_block: Block = self.body.blocks[0]
        block_arg_types = (IndexType(), *iter_types)
        arg_types = entry_block.arg_types
        if block_arg_types != arg_types:
            raise VerifyException(
                "Expected BlockArguments to have the same types as the operands"
            )

    @staticmethod
    def from_region(
        lowerBoundOperands: Sequence[Operation | SSAValue],
        upperBoundOperands: Sequence[Operation | SSAValue],
        inits: Sequence[Operation | SSAValue],
        result_types: Sequence[Attribute],
        lower_bound: int | AffineMapAttr,
        upper_bound: int | AffineMapAttr,
        region: Region,
        step: int | IntegerAttr = 1,
    ) -> ForOp:
        if isinstance(lower_bound, int):
            lower_bound = AffineMapAttr(
                AffineMap(0, 0, (AffineExpr.constant(lower_bound),))
            )
        if isinstance(upper_bound, int):
            upper_bound = AffineMapAttr(
                AffineMap(0, 0, (AffineExpr.constant(upper_bound),))
            )
        if isinstance(step, int):
            step = IntegerAttr.from_index_int_value(step)
        properties: dict[str, Attribute] = {
            "lowerBoundMap": lower_bound,
            "upperBoundMap": upper_bound,
            "step": step,
        }
        return ForOp.build(
            operands=[lowerBoundOperands, upperBoundOperands, inits],
            result_types=[result_types],
            properties=properties,
            regions=[region],
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        unresolved_indvar = parser.parse_argument(expect_type=False)
        parser.parse_characters("=")

        lower_bound_map, lower_bound_operands = _parse_affine_for_bound(parser, "max")
        parser.parse_characters("to")
        upper_bound_map, upper_bound_operands = _parse_affine_for_bound(parser, "min")

        if parser.parse_optional_characters("step") is not None:
            step_pos = parser.pos
            step = parser.parse_integer(allow_boolean=False)
            if step < 0:
                parser.raise_error(
                    "expected step to be representable as a positive signed integer",
                    step_pos,
                )
        else:
            step = 1

        unresolved_iter_args: Sequence[Parser.UnresolvedArgument] = ()
        iter_arg_operands: Sequence[SSAValue] = ()
        iter_arg_types: Sequence[Attribute] = ()

        if parser.parse_optional_characters("iter_args") is not None:

            def parse_iter_arg() -> tuple[Parser.UnresolvedArgument, SSAValue]:
                arg = parser.parse_argument(expect_type=False)
                parser.parse_characters("=")
                return arg, parser.parse_operand()

            pairs = parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, parse_iter_arg
            )
            unresolved_iter_args = tuple(arg for arg, _ in pairs)
            iter_arg_operands = tuple(val for _, val in pairs)
            parser.parse_characters("->")

            # MLIR's `parseArrowTypeList` also accepts a single bare type
            # (no parens) when there is only one loop-carried value.
            if parser.parse_optional_punctuation("(") is not None:
                iter_arg_types = parser.parse_comma_separated_list(
                    Parser.Delimiter.NONE, parser.parse_type
                )
                parser.parse_punctuation(")")
            else:
                iter_arg_types = (parser.parse_type(),)

        iter_args = tuple(
            u_arg.resolve(t) for u_arg, t in zip(unresolved_iter_args, iter_arg_types)
        )
        indvar = unresolved_indvar.resolve(IndexType())
        body = parser.parse_region((indvar, *iter_args))

        # affine.for has no implicit-terminator trait (see TODO above), so the
        # terminator omitted from the printed form when there are no iter_args
        # must be re-inserted by hand here, mirroring `ensureTerminator`
        block = body.block
        if block.last_op is None or not isinstance(block.last_op, YieldOp):
            block.add_op(YieldOp.get())

        attributes = parser.parse_optional_attr_dict()

        for_op = cast(
            Self,
            cls.from_region(
                lower_bound_operands,
                upper_bound_operands,
                iter_arg_operands,
                iter_arg_types,
                lower_bound_map,
                upper_bound_map,
                body,
                step,
            ),
        )
        for_op.attributes |= attributes
        return for_op

    def print(self, printer: Printer):
        printer.print_string(" ")
        indvar, *block_iter_args = self.body.block.args
        printer.print_block_argument(indvar, print_type=False)
        printer.print_string(" = ")

        _print_affine_for_bound(
            printer, self.lowerBoundMap, self.lowerBoundOperands, "max"
        )
        printer.print_string(" to ")
        _print_affine_for_bound(
            printer, self.upperBoundMap, self.upperBoundOperands, "min"
        )

        if self.step.value.data != 1:
            printer.print_string(f" step {self.step.value.data}")

        print_block_terminators = False

        if self.inits:
            printer.print_string(" iter_args")

            with printer.in_parens():
                printer.print_list(
                    zip(block_iter_args, self.inits),
                    lambda pair: print_assignment(printer, *pair),
                )

            printer.print_string(" -> ")

            with printer.in_parens():
                printer.print_list(self.result_types, printer.print_attribute)

            print_block_terminators = True

        printer.print_string(" ")
        printer.print_region(
            self.body,
            print_entry_block_args=False,
            print_empty_block=False,
            print_block_terminators=print_block_terminators,
        )
        printer.print_op_attributes(self.attributes)


@irdl_op_definition
class IfOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affineif-affineaffineifop).
    """

    name = "affine.if"

    args = var_operand_def(IndexType)
    res = var_result_def()

    condition = prop_def(AffineSetAttr)

    then_region = region_def("single_block")
    else_region = region_def()

    traits = traits_def(RecursiveMemoryEffect(), RecursivelySpeculatable())


@irdl_op_definition
class ParallelOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affineparallel-affineaffineparallelop).
    """

    name = "affine.parallel"

    map_operands = var_operand_def(IndexType)

    reductions = prop_def(ArrayAttr[StringAttr])
    lowerBoundsMap = prop_def(AffineMapAttr)
    lowerBoundsGroups = prop_def(DenseIntElementsAttr)
    upperBoundsMap = prop_def(AffineMapAttr)
    upperBoundsGroups = prop_def(DenseIntElementsAttr)
    steps = prop_def(ArrayAttr[IntegerAttr[IntegerType]])

    res = var_result_def()

    body = region_def("single_block")

    def verify_(self) -> None:
        if (
            len(self.operands)
            != len(self.results)
            + self.lowerBoundsMap.data.num_dims
            + self.upperBoundsMap.data.num_dims
            + self.lowerBoundsMap.data.num_symbols
            + self.upperBoundsMap.data.num_symbols
        ):
            raise VerifyException(
                "Expected as many operands as results, lower bound args and upper bound args."
            )

        if sum(self.lowerBoundsGroups.get_values()) != len(
            self.lowerBoundsMap.data.results
        ):
            raise VerifyException("Expected a lower bound group for each lower bound")
        if sum(self.upperBoundsGroups.get_values()) != len(
            self.upperBoundsMap.data.results
        ):
            raise VerifyException("Expected an upper bound group for each upper bound")


_AFFINE_EXPR_PRECEDENCE = {
    AffineBinaryOpKind.Add: 10,
    AffineBinaryOpKind.Mul: 20,
    AffineBinaryOpKind.Mod: 20,
    AffineBinaryOpKind.FloorDiv: 20,
    AffineBinaryOpKind.CeilDiv: 20,
}


def _print_affine_expr_of_ssa_ids(
    printer: Printer,
    expr: AffineExpr,
    operands: Sequence[SSAValue],
    min_prec: int = 0,
) -> None:
    """
    Print an AffineExpr printing `%<name>` in place of each dimension or symbol,
    instead of `d<i>`/`s<i>`.

    Parenthesizes the minimum needed to preserve meaning: a subexpression is wrapped
    if its own precedence is lower than `min_prec`, or, for the right operand of
    `mod`/`floordiv`/`ceildiv` (which aren't associative or commutative), equal to it.

    This is required to match MLIR.
    """
    match expr:
        case AffineConstantExpr(value=value):
            printer.print_string(str(value))
        case AffineDimExpr(position=position):
            printer.print_ssa_value(operands[position])
        case AffineSymExpr(position=position):
            printer.print_ssa_value(operands[position])
        case AffineBinaryOpExpr(kind=kind, lhs=lhs, rhs=rhs):
            prec = _AFFINE_EXPR_PRECEDENCE[kind]
            needs_parens = prec < min_prec

            ctx = printer.in_parens() if needs_parens else nullcontext()

            with ctx:
                _print_affine_expr_of_ssa_ids(printer, lhs, operands, prec)
                printer.print_string(f" {kind.get_token()} ")
                right_min = prec if kind == AffineBinaryOpKind.Add else prec + 1
                _print_affine_expr_of_ssa_ids(printer, rhs, operands, right_min)
        case _:
            raise ValueError(f"Unexpected affine expr {expr}")


def _print_affine_map_of_ssa_ids(
    printer: Printer, map: AffineMap, operands: Sequence[SSAValue]
) -> None:
    """
    Prints `[expr_0, ..., expr_n]`.
    """
    with printer.in_square_brackets():
        printer.print_list(
            map.results,
            lambda res: _print_affine_expr_of_ssa_ids(printer, res, operands),
        )


def _parse_affine_memref_access(
    parser: Parser,
) -> tuple[SSAValue[MemRefType], AffineMap, Sequence[SSAValue[IndexType]], MemRefType]:
    """
    Parses `%memref[<affine-map-of-ssa-ids>] : <type>`.
    """
    memref = parser.parse_unresolved_operand()
    affine_map, indices = parser.parse_affine_map_of_ssa_ids()
    parser.parse_punctuation(":")
    memref_type = parser.parse_type()

    if not isa(memref_type, MemRefType):
        parser.raise_error("Expected memref type")

    memref = parser.resolve_operand(memref, memref_type)

    return memref, affine_map, indices, memref_type


def _print_affine_memref_access(
    printer: Printer,
    memref: SSAValue,
    affine_map: AffineMap,
    indices: Sequence[SSAValue],
    memref_type: Attribute,
) -> None:
    printer.print_ssa_value(memref)
    _print_affine_map_of_ssa_ids(printer, affine_map, indices)
    printer.print_string(" : ")
    printer.print_attribute(memref_type)


def _map_or_identity_attr(
    map: AffineMap | AffineMapAttr | None, accessing: ShapedType
) -> AffineMapAttr:
    """
    Creates an `AffineMapAttr` from a provided `AffineMap`, or the identity
    inferred from the rank of the memref being accessed.
    """
    if isinstance(map, AffineMapAttr):
        return map

    if isinstance(map, AffineMap):
        return AffineMapAttr(map)

    rank = accessing.get_num_dims()
    return AffineMapAttr(AffineMap.identity(rank))


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "affine.store"

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)

    def __init__(
        self,
        value: SSAValue,
        memref: SSAValue,
        indices: Sequence[SSAValue],
        map: AffineMapAttr | None = None,
    ):
        if map is None:
            if not isa(memref_type := memref.type, MemRefType):
                raise ValueError(
                    "affine.store memref operand must be of type MemRefType"
                )

            map = _map_or_identity_attr(map, memref_type)

        super().__init__(
            operands=(value, memref, indices),
            properties={"map": map},
        )

    @classmethod
    def parse(cls, parser: Parser) -> StoreOp:
        value = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        memref, affine_map, indices, memref_type = _parse_affine_memref_access(parser)
        resolved_value = parser.resolve_operand(value, memref_type.get_element_type())
        return StoreOp(resolved_value, memref, indices, AffineMapAttr(affine_map))

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(", ")

        _print_affine_memref_access(
            printer, self.memref, self.map.data, self.indices, self.memref.type
        )


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "affine.load"

    T: ClassVar = VarConstraint("T", AnyAttr())

    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)

    result = result_def(T)

    map = prop_def(AffineMapAttr)

    def __init__(
        self,
        memref: SSAValue,
        indices: Sequence[SSAValue],
        map: AffineMapAttr | None = None,
        result_type: Attribute | None = None,
    ):
        if map is None:
            # Create identity map for memrefs with at least one dimension or () -> ()
            # for zero-dimensional memrefs.
            if not isinstance(memref.type, ShapedType):
                raise ValueError(
                    "affine.load memref operand must be of type ShapedType"
                )

            map = _map_or_identity_attr(map, memref.type)

        if result_type is None:
            if not isa(memref.type, ContainerType):
                raise ValueError(
                    "affine.load memref operand must be of type ContainerType"
                )

            result_type = memref.type.get_element_type()

        super().__init__(
            operands=(memref, indices),
            properties={"map": map},
            result_types=(result_type,),
        )

    @classmethod
    def parse(cls, parser: Parser) -> LoadOp:
        memref, affine_map, indices, memref_type = _parse_affine_memref_access(parser)
        result_type = memref_type.get_element_type()

        return LoadOp(memref, indices, AffineMapAttr(affine_map), result_type)

    def print(self, printer: Printer):
        printer.print_string(" ")
        _print_affine_memref_access(
            printer, self.memref, self.map.data, self.indices, self.memref.type
        )


@irdl_op_definition
class MinOp(IRDLOperation):
    name = "affine.min"
    arguments = var_operand_def(IndexType())
    result = result_def(IndexType())

    map = prop_def(AffineMapAttr)

    def __init__(
        self,
        map_operands: Sequence[SSAValue | Operation],
        affine_map: AffineMapAttr,
    ):
        super().__init__(
            operands=[map_operands],
            properties={"map": affine_map},
            result_types=[IndexType()],
        )

    def verify_(self) -> None:
        if len(self.operands) != self.map.data.num_dims + self.map.data.num_symbols:
            raise VerifyException(
                f"{self.name} expects "
                f"{self.map.data.num_dims + self.map.data.num_symbols} "
                f"operands, but got {len(self.operands)}. The number of map operands "
                "must match the sum of the dimensions and symbols of its map."
            )

    @classmethod
    def parse(cls, parser: Parser) -> MinOp:
        pos = parser.pos
        m = parser.parse_attribute()
        if not isinstance(m, AffineMapAttr):
            parser.raise_error("Expected affine map attr", at_position=pos)
        dims = parser.parse_optional_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parser.parse_operand()
        )
        if dims is None:
            dims = []
        syms = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: parser.parse_operand()
        )
        if syms is None:
            syms = []
        return MinOp(dims + syms, m)

    def print(self, printer: Printer):
        m = self.map.data
        operands = tuple(self.arguments)
        assert len(operands) == m.num_dims + m.num_symbols, f"{len(operands)} {m}"
        printer.print_string(" ")
        printer.print_attribute(self.map)
        printer.print_string(" (")
        if m.num_dims:
            printer.print_list(
                operands[: m.num_dims], lambda el: printer.print_operand(el)
            )
        printer.print_string(")")

        if m.num_symbols:
            printer.print_string("[")
            printer.print_list(
                operands[m.num_dims :], lambda el: printer.print_operand(el)
            )
            printer.print_string("]")


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "affine.yield"
    arguments = var_operand_def()

    traits = traits_def(IsTerminator(), Pure())

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"

    @staticmethod
    def get(*operands: SSAValue | Operation) -> YieldOp:
        return YieldOp.create(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class VectorLoadOp(IRDLOperation):
    """
    Reads a slice from a MemRef into a vector.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affinevector_load-affineaffinevectorloadop).
    """

    name = "affine.vector_load"

    T: ClassVar = VarConstraint("T", AnyAttr())

    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)

    result = result_def(VectorType.constr(T))

    map = prop_def(AffineMapAttr)

    def __init__(
        self,
        memref: SSAValue[MemRefType],
        indices: Sequence[SSAValue[IndexType]],
        map: AffineMap | AffineMapAttr | None = None,
        result_type: VectorType | None = None,
    ):
        map = _map_or_identity_attr(map, memref.type)
        result_type = result_type or VectorType(memref.type.get_element_type(), [])

        super().__init__(
            operands=(memref, indices),
            properties={"map": map},
            result_types=[result_type],
        )

    @classmethod
    def parse(cls, parser: Parser) -> VectorLoadOp:
        memref, affine_map, indices, _ = _parse_affine_memref_access(parser)
        parser.parse_punctuation(",")
        result_type = parser.parse_type()

        if not isa(result_type, VectorType):
            parser.raise_error(
                f"Expected {cls.name} to return a {VectorType.name}, "
                + f"but found: {result_type}"
            )

        return VectorLoadOp(memref, indices, AffineMapAttr(affine_map), result_type)

    def print(self, printer: Printer):
        printer.print_string(" ")
        _print_affine_memref_access(
            printer, self.memref, self.map.data, self.indices, self.memref.type
        )
        printer.print_string(", ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class VectorStoreOp(IRDLOperation):
    """
    Writes a vector into a slice within a MemRef.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Affine/#affinevector_store-affineaffinevectorstoreop).
    """

    name = "affine.vector_store"

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(VectorType.constr(T))
    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)

    map = prop_def(AffineMapAttr)

    def __init__(
        self,
        value: SSAValue,
        memref: SSAValue[MemRefType],
        indices: Sequence[SSAValue[IndexType]],
        map: AffineMap | AffineMapAttr | None = None,
    ):
        map = _map_or_identity_attr(map, memref.type)

        super().__init__(
            operands=(value, memref, indices),
            properties={"map": map},
        )

    @classmethod
    def parse(cls, parser: Parser) -> VectorStoreOp:
        value = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        memref, affine_map, indices, _ = _parse_affine_memref_access(parser)

        parser.parse_punctuation(",")
        value_type = parser.parse_type()

        resolved_value = parser.resolve_operand(value, value_type)
        return VectorStoreOp(resolved_value, memref, indices, AffineMapAttr(affine_map))

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(", ")

        _print_affine_memref_access(
            printer, self.memref, self.map.data, self.indices, self.memref.type
        )

        printer.print_string(", ")
        printer.print_attribute(self.value.type)


Affine = Dialect(
    "affine",
    [
        ApplyOp,
        ForOp,
        ParallelOp,
        IfOp,
        StoreOp,
        LoadOp,
        MinOp,
        YieldOp,
        VectorLoadOp,
        VectorStoreOp,
    ],
    [],
)
