from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, cast

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AffineSetAttr,
    AnyIntegerAttr,
    ArrayAttr,
    ContainerType,
    DenseIntOrFPElementsAttr,
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
    ConstraintVar,
    IRDLOperation,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ApplyOp(IRDLOperation):
    name = "affine.apply"

    mapOperands = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)
    result = result_def(IndexType)

    def __init__(self, map_operands: Sequence[SSAValue], affine_map: AffineMapAttr):
        super().__init__(
            operands=[map_operands],
            properties={"map": affine_map},
            result_types=[IndexType()],
        )

    def verify_(self) -> None:
        if len(self.mapOperands) != self.map.data.num_dims + self.map.data.num_symbols:
            raise VerifyException(
                f"{self.name} expects {self.map.data.num_dims + self.map.data.num_symbols} operands, but got {len(self.mapOperands)}. The number of map operands must match the sum of the dimensions and symbols of its map."
            )
        if len(self.map.data.results) != 1:
            raise VerifyException("affine.apply expects a unidimensional map.")


@irdl_op_definition
class For(IRDLOperation):
    name = "affine.for"

    lowerBoundOperands: VarOperand = var_operand_def(IndexType)
    upperBoundOperands: VarOperand = var_operand_def(IndexType)
    inits: VarOperand = var_operand_def()
    res: VarOpResult = var_result_def(AnyAttr())

    lowerBoundMap = prop_def(AffineMapAttr)
    upperBoundMap = prop_def(AffineMapAttr)
    step: AnyIntegerAttr = prop_def(AnyIntegerAttr)

    body: Region = region_def()

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
        iter_types = [op.type for op in self.inits]
        if iter_types != [res.type for res in self.results]:
            raise VerifyException(
                "Expected all operands and result pairs to have matching types"
            )
        entry_block: Block = self.body.blocks[0]
        block_arg_types = [IndexType()] + iter_types
        arg_types = [arg.type for arg in entry_block.args]
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
        step: int | AnyIntegerAttr = 1,
    ) -> For:
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
        return For.build(
            operands=[lowerBoundOperands, upperBoundOperands, inits],
            result_types=[result_types],
            properties=properties,
            regions=[region],
        )


@irdl_op_definition
class If(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Affine/#affineif-affineaffineifop
    """

    name = "affine.if"

    args = var_operand_def(IndexType)
    res = var_result_def()

    condition = attr_def(AffineSetAttr)

    then_region = region_def("single_block")
    else_region = region_def()


@irdl_op_definition
class ParallelOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Affine/#affineparallel-affineaffineparallelop
    """

    name = "affine.parallel"

    map_operands = var_operand_def(IndexType)

    reductions = prop_def(ArrayAttr[StringAttr])
    lowerBoundsMap = prop_def(AffineMapAttr)
    lowerBoundsGroups = prop_def(DenseIntOrFPElementsAttr)
    upperBoundsMap = prop_def(AffineMapAttr)
    upperBoundsGroups = prop_def(DenseIntOrFPElementsAttr)
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

        if sum(g.value.data for g in self.lowerBoundsGroups.data) != len(
            self.lowerBoundsMap.data.results
        ):
            raise VerifyException("Expected a lower bound group for each lower bound")
        if sum(g.value.data for g in self.upperBoundsGroups.data) != len(
            self.upperBoundsMap.data.results
        ):
            raise VerifyException("Expected an upper bound group for each upper bound")


@irdl_op_definition
class Store(IRDLOperation):
    name = "affine.store"

    T = Annotated[Attribute, ConstraintVar("T")]

    value = operand_def(T)
    memref = operand_def(MemRefType[T])
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
                    "affine.store memref operand must be of type MemrefType"
                )
            rank = memref_type.get_num_dims()
            map = AffineMapAttr(AffineMap.identity(rank))
        super().__init__(
            operands=(value, memref, indices),
            properties={"map": map},
        )


@irdl_op_definition
class Load(IRDLOperation):
    name = "affine.load"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    indices = var_operand_def(IndexType)

    result = result_def(T)

    map = opt_prop_def(AffineMapAttr)

    def __init__(
        self,
        memref: SSAValue,
        indices: Sequence[SSAValue],
        map: AffineMapAttr | None = None,
        result_type: T | None = None,
    ):
        if map is None:
            # Create identity map for memrefs with at least one dimension or () -> ()
            # for zero-dimensional memrefs.
            if not isinstance(memref.type, ShapedType):
                raise ValueError(
                    "affine.store memref operand must be of type ShapedType"
                )
            memref_type = cast(MemRefType[Attribute], memref.type)
            rank = memref_type.get_num_dims()
            map = AffineMapAttr(AffineMap.identity(rank))
        if result_type is None:
            # Create identity map for memrefs with at least one dimension or () -> ()
            # for zero-dimensional memrefs.
            if not isinstance(memref.type, ContainerType):
                raise ValueError(
                    "affine.store memref operand must be of type ContainerType"
                )
            memref_type = cast(ContainerType[Any], memref.type)
            result_type = memref_type.get_element_type()

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
                f"{self.name} expects {self.map.data.num_dims + self.map.data.num_symbols} operands, but got {len(self.operands)}. The number of map operands must match the sum of the dimensions and symbols of its map."
            )


@irdl_op_definition
class Yield(IRDLOperation):
    name = "affine.yield"
    arguments: VarOperand = var_operand_def(AnyAttr())

    traits = frozenset([IsTerminator()])

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(operands=[SSAValue.get(operand) for operand in operands])


Affine = Dialect(
    "affine",
    [
        ApplyOp,
        For,
        ParallelOp,
        If,
        Store,
        Load,
        MinOp,
        Yield,
    ],
    [],
)
