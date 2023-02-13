from __future__ import annotations

from xdsl.ir import OpResult, SSAValue, Dialect
from xdsl.ir import ParametrizedAttribute

from xdsl.irdl import (irdl_op_definition, Operation, Operand, Annotated,
                       irdl_attr_definition, AnyAttr)


@irdl_attr_definition
class MPIRequest(ParametrizedAttribute):
    name = "MPI_Request"


@irdl_attr_definition
class MPIStatus(ParametrizedAttribute):
    name = "MPI_Status"


@irdl_op_definition
class MPIRequestOp(Operation):
    # Name of the type. This is used for printing and parsing.
    name: str = "mpi.MPIRequestOp"
    request: Annotated[Operand, AnyAttr()]
    # output: Annotated[OpResult, Attribute]

    @staticmethod
    def from_attr(request: AnyAttr) -> MPIRequestOp:
        return MPIRequestOp.create(operands=[], result_types=[request])


@irdl_op_definition
class MPI_Wait(Operation):
    name: str = "mpi.MPI_Wait"

    request: Annotated[Operand, AnyAttr()]
    status: Annotated[OpResult, MPIStatus()]

    @staticmethod
    def from_callable(request: Operation) -> MPI_Wait:
        op1 = SSAValue.get(request)
        return MPI_Wait.build(operands=[op1], result_types=[MPIStatus()])


MPI = Dialect([MPI_Wait, MPIRequestOp], [MPIRequest, MPIStatus])
