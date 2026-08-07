from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import ClassVar

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

    def verify_(self) -> None:
        if len(self.operands) != self.map.data.num_dims + self.map.data.num_symbols:
            raise VerifyException(
                f"{self.name} expects "
                f"{self.map.data.num_dims + self.map.data.num_symbols} "
                "operands, but got {len(self.operands)}. The number of map operands "
                "must match the sum of the dimensions and symbols of its map."
            )


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "affine.yield"
    arguments = var_operand_def()

    traits = traits_def(IsTerminator(), Pure())

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
