#!/usr/bin/env python3
"""Microbenchmark properties of the xDSL implementation."""

from __future__ import annotations

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
)


@irdl_op_definition
class EmptyOp(IRDLOperation):
    """An empty operation."""

    name = "empty"


class OpCreation:
    """Benchmark creating an operation in xDSL."""

    def time_operation_build(self) -> None:
        """Time building an empty operation.

        For comparison with the "How Slow is MLIR" testbench
        `CreateOps/simpleRegistered`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b(ctx.get());
        for (auto _ : state) {
            for (int j = 0; j < state.range(0); ++j) {
            b.create<EmptyOp>(unknownLoc);
            }
        }
        state.SetComplexityN(state.range(0));
        ```
        """
        EmptyOp()

    def time_operation_create_optimised(self) -> None:
        """Time creating an empty operation directly."""
        empty_op = EmptyOp.__new__(EmptyOp)
        empty_op._operands = ()  # pyright: ignore[reportPrivateUsage]
        empty_op.results = ()
        empty_op.properties = {}
        empty_op.attributes = {}
        empty_op._successors = ()  # pyright: ignore[reportPrivateUsage]
        empty_op.regions = ()


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    OP_CREATION = OpCreation()
    profile(
        {
            "OpCreation.operation_build": Benchmark(OP_CREATION.time_operation_build),
            "OpCreation.operation_create_optimised": Benchmark(
                OP_CREATION.time_operation_create_optimised
            ),
        }
    )
