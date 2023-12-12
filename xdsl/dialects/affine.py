from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, cast

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AffineSetAttr,
    AnyIntegerAttr,
    ContainerType,
    IndexType,
    IntegerAttr,
    ShapedType,
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.irdl import (
    AnyAttr,
    ConstraintVar,
    IRDLOperation,
    VarOperand,
    VarOpResult,
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

    arguments: VarOperand = var_operand_def(AnyAttr())
    res: VarOpResult = var_result_def(AnyAttr())

    lower_bound = prop_def(AffineMapAttr)
    upper_bound = prop_def(AffineMapAttr)
    step: AnyIntegerAttr = prop_def(AnyIntegerAttr)

    body: Region = region_def()

    # TODO this requires the ImplicitAffineTerminator trait instead of
    # NoTerminator
    # gh issue: https://github.com/xdslproject/xdsl/issues/1149

    def verify_(self) -> None:
        if (
            len(self.operands)
            != len(self.results)
            + self.lower_bound.data.num_dims
            + self.upper_bound.data.num_dims
            + self.lower_bound.data.num_symbols
            + self.upper_bound.data.num_symbols
        ):
            raise VerifyException(
                "Expected as many operands as results, lower bound args and upper bound args."
            )

        iter_types = [op.type for op in self.operands[-len(self.results) :]]
        if iter_types != [res.type for res in self.results]:
            raise VerifyException(
                "Expected all operands and result pairs to have matching types"
            )
        if any(op.type != IndexType() for op in self.operands[: -len(self.results)]):
            raise VerifyException("Expected all bounds arguments types to be index")

        entry_block: Block = self.body.blocks[0]
        block_arg_types = [IndexType()] + iter_types
        arg_types = [arg.type for arg in entry_block.args]
        if block_arg_types != arg_types:
            raise VerifyException(
                "Expected BlockArguments to have the same types as the operands"
            )

    @staticmethod
    def from_region(
        operands: Sequence[Operation | SSAValue],
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
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }
        return For.build(
            operands=[operands],
            result_types=[result_types],
            properties=properties,
            regions=[region],
        )


@irdl_op_definition
class If(IRDLOperation):
    name = "affine.if"

    args = var_operand_def(IndexType)
    res = var_result_def()

    condition = prop_def(AffineSetAttr)

    then_region = region_def("single_block")
    else_region = region_def()


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
        If,
        Store,
        Load,
        MinOp,
        Yield,
    ],
    [],
)
