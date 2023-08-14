from __future__ import annotations

from abc import ABC
from typing import cast

from xdsl.dialects import llvm
from xdsl.dialects.builtin import AnyFloat, IntegerType, Signedness, i32
from xdsl.dialects.memref import MemRefType
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_result_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer

t_bool: IntegerType = IntegerType(1, Signedness.SIGNLESS)

AnyNumericType = AnyFloat | IntegerType


@irdl_attr_definition
class DataType(ParametrizedAttribute, TypeAttribute):
    """
    This type represents MPI_Datatype
    """

    name = "mpi.datatype"


class MPIBaseOp(IRDLOperation, ABC):
    """
    Base class for MPI Operations
    """

    pass


@irdl_op_definition
class SendOp(MPIBaseOp):
    """
    This wraps the MPI_Send function (blocking send)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Send.html

    ## The MPI_Send Function Docs:

    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)

        - buf: Initial address of send buffer (choice).
        - count: Number of elements in send buffer (non-negative integer).
        - datatype: Datatype of each send buffer element (handle).
        - dest: Rank of destination (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).

    ## Our Abstraction:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.send"

    buffer: Operand = operand_def(Attribute)
    count: Operand = operand_def(i32)
    datatype: Operand = operand_def(DataType)
    dest: Operand = operand_def(i32)
    tag: Operand = operand_def(i32)

    # omitted for simplicity
    # status : OpResult = result_def(i32)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        dest: SSAValue | Operation,
        tag: SSAValue | Operation,
    ):
        super().__init__(
            operands=[buffer, count, datatype, dest, tag],
            result_types=[],
        )


@irdl_op_definition
class RecvOp(MPIBaseOp):
    """
    This wraps the MPI_Recv function (blocking receive).
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Recv.html

    ## The MPI_Recv Function Docs:

    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)

        - buf: Initial address of receive buffer (choice).
        - count: Number of elements in receive buffer (integer).
        - datatype: Datatype of each receive buffer element (handle).
        - source: Rank of source (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).
        - status: status object (Status).

    ## Our Abstractions:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.recv"

    buffer: Operand = operand_def(Attribute)
    count: Operand = operand_def(i32)
    datatype: Operand = operand_def(DataType)
    source: Operand = operand_def(i32)
    tag: Operand = operand_def(i32)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        source: SSAValue | Operation,
        tag: SSAValue | Operation,
    ):
        super().__init__(
            operands=[buffer, count, datatype, source, tag],
            result_types=[],
        )


@irdl_op_definition
class CommRankOp(MPIBaseOp):
    """
    Represents the MPI_Comm_size(MPI_Comm comm, int *rank) function call which returns
    the rank of the communicator

    Currently limited to COMM_WORLD
    """

    name = "mpi.comm.rank"

    rank: OpResult = result_def(i32)

    def __init__(self):
        super().__init__(result_types=[i32])

    def print(self, printer: Printer):
        if self.attributes:
            printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print(self.rank.type)

    @classmethod
    def parse(cls, parser: Parser) -> CommRankOp:
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        res_type = parser.parse_type()
        assert res_type == i32
        op = cls()
        op.attributes.update(attrs)
        return op


@irdl_op_definition
class CommSizeOp(MPIBaseOp):
    """
    Represents the MPI_Comm_size(MPI_Comm comm, int *size) function call which returns
    the size of the communicator

    Currently limited to COMM_WORLD
    """

    name = "mpi.comm.size"

    size: OpResult = result_def(i32)

    def __init__(self):
        super().__init__(result_types=[i32])

    def print(self, printer: Printer):
        if self.attributes:
            printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print(self.size.type)

    @classmethod
    def parse(cls, parser: Parser) -> CommSizeOp:
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        res_type = parser.parse_type()
        assert res_type == i32
        op = cls()
        op.attributes.update(attrs)
        return op


@irdl_op_definition
class InitOp(MPIBaseOp):
    """
    This represents a bare MPI_Init call with both args being nullptr
    """

    name = "mpi.init"

    status = opt_result_def(i32)

    def __init__(self, /, check_status: bool = False):
        super().__init__(result_types=[[i32] if check_status else []])

    def print(self, printer: Printer):
        if self.attributes:
            printer.print_op_attributes(self.attributes)
        if self.status:
            printer.print_string(" : ")
            printer.print(self.status.type)

    @classmethod
    def parse(cls, parser: Parser) -> InitOp:
        attrs = parser.parse_optional_attr_dict()
        if parser.parse_optional_punctuation(":"):
            res_type = parser.parse_type()
            assert res_type == i32
            op = cls(check_status=True)
        else:
            op = cls()
        op.attributes.update(attrs)
        return op


@irdl_op_definition
class FinalizeOp(MPIBaseOp):
    """
    This represents an MPI_Finalize call with both args being nullptr
    """

    name = "mpi.finalize"

    def __init__(self):
        super().__init__()

    def print(self, printer: Printer):
        if self.attributes:
            printer.print_op_attributes(self.attributes)

    @classmethod
    def parse(cls, parser: Parser) -> FinalizeOp:
        attrs = parser.parse_optional_attr_dict()
        op = cls()
        op.attributes.update(attrs)
        return op


@irdl_op_definition
class UnwrapMemrefOp(MPIBaseOp):
    """
    This Op can be used as a helper to get memrefs into MPI calls.

    It takes any MemRef as input, and returns an llvm.ptr, element count and MPI_Datatype.
    """

    name = "mpi.mlir.unwrap_memref"

    ref: Operand = operand_def(MemRefType[AnyNumericType])

    ptr: OpResult = result_def(llvm.LLVMPointerType)
    len: OpResult = result_def(i32)
    type: OpResult = result_def(DataType)

    def __init__(self, ref: SSAValue | Operation):
        ssa_val = SSAValue.get(ref)
        assert isinstance(ssa_val.type, MemRefType)
        elem_type = cast(MemRefType[AnyNumericType], ssa_val.type).element_type

        super().__init__(
            operands=[ref],
            result_types=[llvm.LLVMPointerType.typed(elem_type), i32, DataType()],
        )


@irdl_op_definition
class GetDtypeOp(MPIBaseOp):
    """
    This op is used to convert MLIR types to MPI_Datatype constants.

    So, e.g. if you want to get the `MPI_Datatype` for an `i32` you can use

        %dtype = mpi.get_dtype of "dtype" = i32} : () -> mpi.datatype

    to get the magic constant. See `_MPIToLLVMRewriteBase._translate_to_mpi_type`
    docstring for more detail on which types are supported.
    """

    name = "mpi.mlir.get_dtype"

    dtype: Attribute = attr_def(Attribute)

    result: OpResult = result_def(DataType)

    def __init__(self, dtype: Attribute):
        super().__init__(result_types=[DataType()], attributes={"dtype": dtype})

    def print(self, printer: Printer):
        attrs = self.attributes.copy()
        dtype: Attribute = attrs.pop("dtype")
        if attrs:
            printer.print_op_attributes(attrs)

        printer.print(" of ")
        printer.print(dtype)
        printer.print(" -> ")
        printer.print(self.result.type)

    @classmethod
    def parse(cls, parser: Parser) -> GetDtypeOp:
        attrs: dict[str, Attribute] = dict()
        if not parser.parse_optional_characters("of"):
            attrs = parser.parse_optional_attr_dict()
            parser.parse_characters("of")

        dtype = parser.parse_attribute()
        parser.parse_punctuation("->")
        res_typ = parser.parse_attribute()
        assert isinstance(res_typ, DataType)
        op = cls(dtype)
        op.attributes.update(attrs)
        return op


MPI = Dialect(
    [
        InitOp,
        CommRankOp,
        UnwrapMemrefOp,
        GetDtypeOp,
        SendOp,
        RecvOp,
        FinalizeOp,
    ],
    [
        DataType,
    ],
)
