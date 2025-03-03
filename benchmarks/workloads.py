#!/usr/bin/env python3
"""Workloads for benchmarking xDSL."""

import random
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent
EXTRA_MLIR_DIR = BENCHMARKS_DIR / "resources" / "extra_mlir"


class WorkloadBuilder:
    """A helper class to programmatically build synthetic workloads."""

    @classmethod
    def wrap_module(cls, ops: list[str]) -> str:
        """Wrap a list of operations as a module."""
        workload = f'"builtin.module"() ({{\n  {"\n  ".join(ops)}\n}}) : () -> ()'
        return workload

    @classmethod
    def empty(cls) -> str:
        """Generate an empty module."""
        return WorkloadBuilder.wrap_module([])

    @classmethod
    def constant_folding(cls, size: int = 100) -> str:
        """Generate a constant folding workload of a given size."""
        assert size >= 0
        random.seed(0)
        ops: list[str] = []
        ops.append(
            '%0 = "arith.constant"() {"value" = '
            f"{random.randint(1, 1000)} : i32}} : () -> i32"
        )
        for i in range(1, size + 1):
            if i % 2 == 0:
                ops.append(
                    f'%{i} = "arith.addi"(%{i - 1}, %{i - 2}) : (i32, i32) -> i32'
                )
            else:
                ops.append(
                    f'%{i} = "arith.constant"() {{"value" = '
                    f"{random.randint(1, 1000)} : i32}} : () -> i32"
                )
        ops.append(f'"test.op"(%{(size // 2) * 2}) : (i32) -> ()')
        return WorkloadBuilder.wrap_module(ops)

    @classmethod
    def fmadd(cls, size: int = 4) -> str:
        """."""
        assert size >= 0
        random.seed(0)
        ops: list[str] = []
        ops.append(
            '%0 = "arith.constant"() {"value" = '
            f"{random.randint(1, 1000)} : i32}} : () -> i32"
        )
        for i in range(1, size + 1):
            if i % 4 == 0:
                ops.append(
                    f'%{i} = "arith.addi"(%{i - 1}, %{i - 2}) : (i32, i32) -> i32'
                )
            elif i % 4 == 3:
                ops.append(
                    f'%{i} = "arith.muli"(%{i - 2}, %{i - 3}) : (i32, i32) -> i32'
                )
            else:
                ops.append(
                    f'%{i} = "arith.constant"() {{"value" = '
                    f"{random.randint(1, 1000)} : i32}} : () -> i32"
                )
        ops.append(f'"test.op"(%{(size // 4) * 4}) : (i32) -> ()')
        return WorkloadBuilder.wrap_module(ops)

    @classmethod
    def extra_mlir_file(cls, name: str) -> str:
        """."""
        return (EXTRA_MLIR_DIR / name).read_text()

    @classmethod
    def large_dense_attr(cls) -> str:
        """."""
        return WorkloadBuilder.extra_mlir_file("large_dense_attr.mlir")

    @classmethod
    def large_dense_attr_hex(cls) -> str:
        """."""
        return WorkloadBuilder.extra_mlir_file("large_dense_attr_hex.mlir")
