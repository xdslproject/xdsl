"""
The OpenACC (acc) dialect that models the OpenACC programming model in MLIR.

OpenACC is a directive-based programming model for accelerating applications
on heterogeneous systems. This dialect exposes compute constructs, data
constructs, loops, and the associated clauses so that host and accelerator
code can be represented, analysed, and lowered to target-specific runtimes.

See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/).
"""

from xdsl.dialects.builtin import I1, IndexType, IntegerType
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import Attribute, Dialect
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    base,
    irdl_op_definition,
    lazy_traits_def,
    opt_operand_def,
    region_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)

_IntOrIndex = base(IntegerType) | base(IndexType)


@irdl_op_definition
class ParallelOp(IRDLOperation):
    """
    Implementation of upstream acc.parallel.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accparallel-accparallelop).
    """

    name = "acc.parallel"

    async_operands = var_operand_def(_IntOrIndex)
    wait_operands = var_operand_def(_IntOrIndex)
    num_gangs = var_operand_def(_IntOrIndex)
    num_workers = var_operand_def(_IntOrIndex)
    vector_length = var_operand_def(_IntOrIndex)
    if_cond = opt_operand_def(I1)
    self_cond = opt_operand_def(I1)
    reduction_operands = var_operand_def()
    private_operands = var_operand_def()
    firstprivate_operands = var_operand_def()
    data_clause_operands = var_operand_def()

    region = region_def("single_block")

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(YieldOp),
            RecursiveMemoryEffect(),
        )
    )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    """
    Implementation of upstream acc.yield.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accyield-accyieldop).
    """

    name = "acc.yield"

    traits = traits_def(
        IsTerminator(),
        NoMemoryEffect(),
        HasParent(ParallelOp),
    )


ACC = Dialect("acc", [ParallelOp, YieldOp,], [])
