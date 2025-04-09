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
        """Generate a constant folding workload of a given size.

        An example of running `WorkloadBuilder().constant_folding(size=5)`
        is as follows:

        ```mlir
        "builtin.module"() ({
            %0 = "arith.constant"() {"value" = 865 : i32} : () -> i32
            %1 = "arith.constant"() {"value" = 395 : i32} : () -> i32
            %2 = "arith.addi"(%1, %0) : (i32, i32) -> i32
            %3 = "arith.constant"() {"value" = 777 : i32} : () -> i32
            %4 = "arith.addi"(%3, %2) : (i32, i32) -> i32
            %5 = "arith.constant"() {"value" = 912 : i32} : () -> i32
            "test.op"(%4) : (i32) -> ()
        }) : () -> ()
        ```
        """
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
        """Get the MLIR text representation of a large dense attr.

        An example of running `WorkloadBuilder().large_dense_attr(x=3, y=3)`
        is as follows:

        ```mlir
        "builtin.module"() ({
            %0 = "arith.constant"() <{value = dense<[
                [69, 87, -108], [4, 120, 79], [27, 116, 55]
            ]> : tensor<3x3xi8>}> : () -> tensor<3x3xi8>
        }) : () -> ()
        """
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
        """Get the MLIR hex representation of a large dense attr.

        An example of running `WorkloadBuilder().large_dense_attr(x=3, y=3)`
        is as follows:

        ```mlir
        "builtin.module"() ({
            %0 = "arith.constant"() <{
                value = dense<0xCD18FC9FB649438493> : tensor<3x3xi8>
            }> : () -> tensor<3x3xi8>
        }) : () -> ()
        """
        assert x >= 0
        assert y >= 0
        random.seed(RANDOM_SEED)
        # In order to guarantee the hex value encodes a valid i8 tensor without
        # significant logic coupled to the hex implementation, we set all values
        # to zero. Each dense attr item is a byte is 2 hex chars, so we need
        # x * y * 2 characters.
        dense_attr_hex = "0" * x * y * 2
        ops = [
            (
                '%0 = "arith.constant"() '
                f"<{{value = dense<0x{dense_attr_hex}> "
                f": tensor<{x}x{y}xi8>}}> : () -> tensor<{x}x{y}xi8>"
            )
        ]
        return WorkloadBuilder.wrap_module(ops)
