from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, cast

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
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    VarConstraint,
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

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "affine.store"

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)
    map = opt_prop_def(AffineMapAttr)

    def __init__(
        self,
        value: SSAValue,
        memref: SSAValue,
        indices: Sequence[SSAValue],
        map: AffineMapAttr | None = None,
    ):
        if map is None:
            # Create identity map for memrefs with at least one dimension or () -> ()
            # for zero-dimensional memrefs.
            if not isinstance(memref_type := memref.type, MemRefType):
                raise ValueError(
                    "affine.store memref operand must be of type MemRefType"
                )
            rank = memref_type.get_num_dims()
            map = AffineMapAttr(AffineMap.identity(rank))
        super().__init__(
            operands=(value, memref, indices),
            properties={"map": map},
        )


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "affine.load"

    T: ClassVar = VarConstraint("T", AnyAttr())

    memref = operand_def(MemRefType.constr(T))
    indices = var_operand_def(IndexType)

    result = result_def(T)

    map = opt_prop_def(AffineMapAttr)

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
                    "affine.store memref operand must be of type ShapedType"
                )
            memref_type = cast(MemRefType, memref.type)
            rank = memref_type.get_num_dims()
            map = AffineMapAttr(AffineMap.identity(rank))
        if result_type is None:
            # Create identity map for memrefs with at least one dimension or () -> ()
            # for zero-dimensional memrefs.
            if not isa(memref.type, ContainerType):
                raise ValueError(
                    "affine.store memref operand must be of type ContainerType"
                )

            result_type = memref.type.get_element_type()

        super().__init__(
            operands=(memref, indices),
            properties={"map": map},
            result_types=(result_type,),
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
    ],
    [],
)
