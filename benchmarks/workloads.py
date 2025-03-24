#!/usr/bin/env python3
"""Workloads for benchmarking xDSL."""

import random

RANDOM_SEED = 0
HEX_CHARS = "0123456789ABCDEF"


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
        random.seed(RANDOM_SEED)
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
    def large_dense_attr(cls, x: int = 1024, y: int = 1024) -> str:
        """Get the MLIR text representation of a large dense attr."""
        assert x >= 0
        assert y >= 0
        random.seed(RANDOM_SEED)
        dense_attr = [[random.randint(-128, 128) for _ in range(x)] for _ in range(y)]
        ops = [
            (
                '%0 = "arith.constant"() '
                f"<{{value = dense<{dense_attr}> "
                f": tensor<{x}x{y}xi8>}}> : () -> tensor<{x}x{y}xi8>"
            )
        ]
        return WorkloadBuilder.wrap_module(ops)

    @classmethod
    def large_dense_attr_hex(cls, x: int = 1024, y: int = 1024) -> str:
        """Get the MLIR hex representation of a large dense attr."""
        assert x >= 0
        assert y >= 0
        random.seed(RANDOM_SEED)
        # Each dense attr item is a byte = 2 hex chars
        dense_attr_hex = "".join(random.choice(HEX_CHARS) for _ in range(x * y * 2))
        ops = [
            (
                '%0 = "arith.constant"() '
                f"<{{value = dense<0x{dense_attr_hex}> "
                f": tensor<{x}x{y}xi8>}}> : () -> tensor<{x}x{y}xi8>"
            )
        ]
        return WorkloadBuilder.wrap_module(ops)
