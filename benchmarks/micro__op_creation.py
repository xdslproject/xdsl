#!/usr/bin/env python3
"""Benchmark rewriting in xDSL."""

from __future__ import annotations

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, Float64Type, TensorType, f64
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """Constant operation turns a literal into an SSA value."""

    name = "toy.constant"
    value = attr_def(DenseIntOrFPElementsAttr)
    res = result_def(TensorType[Float64Type])

    def __init__(self, value: DenseIntOrFPElementsAttr):
        super().__init__(result_types=[value.type], attributes={"value": value})

    @staticmethod
    def from_list(data: list[float], shape: list[int]) -> ConstantOp:
        value = DenseIntOrFPElementsAttr.tensor_from_list(data, f64, shape)
        return ConstantOp(value)

    def verify_(self) -> None:
        if not self.res.type == self.value.type:
            raise VerifyException(
                "Expected value and result types to be equal: "
                f"{self.res.type}, {self.value.type}"
            )


CONSTANT_OPERATION_X_SIZE = 6
CONSTANT_OPERATION_Y_SIZE = 6
CONSTANT_OPERATION = ConstantOp.from_list(
    [x for x in range(CONSTANT_OPERATION_X_SIZE)],
    [y for y in range(CONSTANT_OPERATION_Y_SIZE)],
)


def time_op_creation__create() -> None:
    """Time creating an operation."""
    ConstantOp.from_list(
        [x for x in range(CONSTANT_OPERATION_X_SIZE)],
        [y for y in range(CONSTANT_OPERATION_Y_SIZE)],
    )


def time_op_creation__clone() -> None:
    """Time cloning an operation."""
    CONSTANT_OPERATION.clone()


if __name__ == "__main__":
    from bench_utils import profile  # type: ignore

    BENCHMARKS = {
        "time_op_creation__create": time_op_creation__create,
        "time_op_creation__clone": time_op_creation__clone,
    }
    profile(BENCHMARKS)
