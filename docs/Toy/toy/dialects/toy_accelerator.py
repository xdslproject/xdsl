from __future__ import annotations

import abc
from typing import Annotated

from xdsl.dialects.builtin import AnyIntegerAttr, ArrayAttr
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class Transpose(IRDLOperation):
    name = "toy_accelerator.transpose"

    destination = operand_def(MemRefType)
    source = operand_def(MemRefType)

    source_rows = attr_def(AnyIntegerAttr)
    source_cols = attr_def(AnyIntegerAttr)

    def __init__(
        self,
        destination: SSAValue | Operation,
        input: SSAValue | Operation,
        source_rows: AnyIntegerAttr,
        source_cols: AnyIntegerAttr,
    ):
        super().__init__(
            operands=(destination, input),
            attributes={
                "source_rows": source_rows,
                "source_cols": source_cols,
            },
        )

    def verify_(self) -> None:
        if not isinstance(self.source.type, MemRefType):
            raise VerifyException(
                f"Invalid transpose source type {self.source.type}, expected MemRefType"
            )
        if not isinstance(self.destination.type, MemRefType):
            raise VerifyException(
                f"Invalid transpose destination type {self.destination.type}, expected MemRefType"
            )

        expected_source_shape = ArrayAttr((self.source_rows, self.source_cols))
        source_shape = self.source.type.shape

        if source_shape != expected_source_shape:
            raise VerifyException("Transpose source shape mismatch")

        expected_destination_shape = ArrayAttr((self.source_cols, self.source_rows))
        destination_shape = self.destination.type.shape

        if destination_shape != expected_destination_shape:
            raise VerifyException("Transpose source shape mismatch")


class BinOp(IRDLOperation, abc.ABC):
    """
    An in-place mutating binary operation.
    """

    T = Annotated[MemRefType[Attribute], ConstraintVar("T")]

    dest = operand_def(T)
    lhs = operand_def(T)
    rhs = operand_def(T)

    def __init__(
        self,
        dest: SSAValue | Operation,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        super().__init__(operands=(dest, lhs, rhs))


@irdl_op_definition
class Add(BinOp):
    name = "toy_accelerator.add"


@irdl_op_definition
class Mul(BinOp):
    name = "toy_accelerator.mul"


ToyAccelerator = Dialect(
    [
        Transpose,
        Add,
        Mul,
    ],
    [],
)
