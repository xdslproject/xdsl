from __future__ import annotations

from typing import Annotated, Any, Sequence, cast

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyIntegerAttr,
    ContainerType,
    IndexType,
    IntegerAttr,
    ShapedType,
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.ir.affine.affine_expr import AffineExpr
from xdsl.ir.affine.affine_map import AffineMap
from xdsl.irdl import (
    AnyAttr,
    ConstraintVar,
    IRDLOperation,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class For(IRDLOperation):
    name = "affine.for"

    arguments: VarOperand = var_operand_def(AnyAttr())
    res: VarOpResult = var_result_def(AnyAttr())

    lower_bound = attr_def(AffineMapAttr)
    upper_bound = attr_def(AffineMapAttr)
    step: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    body: Region = region_def()

    # TODO this requires the ImplicitAffineTerminator trait instead of
    # NoTerminator
    # gh issue: https://github.com/xdslproject/xdsl/issues/1149

    def verify_(self) -> None:
        if len(self.operands) != len(self.results):
            raise Exception("Expected the same amount of operands and results")

        operand_types = [SSAValue.get(op).type for op in self.operands]
        if operand_types != [res.type for res in self.results]:
            raise Exception(
                "Expected all operands and result pairs to have matching types"
            )

        entry_block: Block = self.body.blocks[0]
        block_arg_types = [IndexType()] + operand_types
        arg_types = [arg.type for arg in entry_block.args]
        if block_arg_types != arg_types:
            raise Exception(
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
        attributes: dict[str, Attribute] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }
        return For.build(
            operands=[operands],
            result_types=[result_types],
            attributes=attributes,
            regions=[region],
        )


@irdl_op_definition
class Store(IRDLOperation):
    name = "affine.store"

    T = Annotated[Attribute, ConstraintVar("T")]

    value = operand_def(T)
    memref = operand_def(MemRefType[T])
    indices = var_operand_def(IndexType)
    map = opt_attr_def(AffineMapAttr)

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
            if not isinstance(memref.type, MemRefType):
                raise ValueError(
                    "affine.store memref operand must be of type MemrefType"
                )
            memref_type = cast(MemRefType[Attribute], memref.type)
            rank = memref_type.get_num_dims()
            map = AffineMapAttr(AffineMap.identity(rank))
        super().__init__(
            operands=(value, memref, indices),
            attributes={"map": map},
        )


@irdl_op_definition
class Load(IRDLOperation):
    name = "affine.load"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    indices = var_operand_def(IndexType)

    result = result_def(T)

    map = opt_attr_def(AffineMapAttr)

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
            attributes={"map": map},
            result_types=(result_type,),
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
    [
        For,
        Store,
        Load,
        Yield,
    ],
    [],
)
