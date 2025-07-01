#!/usr/bin/env python3
"""Workloads for benchmarking xDSL."""

import random

from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    ModuleOp,
    TensorType,
    i8,
    i32,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Operation

RANDOM_SEED = 0
HEX_CHARS = "0123456789ABCDEF"


class WorkloadBuilder:
    """A helper class to programmatically build synthetic workloads."""

    @classmethod
    def wrap_module(cls, ops: list[str]) -> str:
        """Wrap a list of operations as a module."""
        joined_ops = "\n  ".join(ops)
        workload = f'"builtin.module"() ({{\n  {joined_ops}\n}}) : () -> ()'
        return workload

    @classmethod
    def empty(cls) -> str:
        """Generate an empty module."""
        return WorkloadBuilder.wrap_module([])

    @classmethod
    def constant_folding_module(cls, size: int = 100) -> ModuleOp:
        """Generate a constant folding workload of a given size.

        The output of running the command
        `print(WorkloadBuilder().constant_folding_module(size=5))` is shown
        below:

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
        ops: list[Operation] = []
        ops.append(ConstantOp(IntegerAttr(random.randint(1, 1000), i32)))
        for i in range(1, size + 1):
            if i % 2 == 0:
                ops.append(AddiOp(ops[i - 1], ops[i - 2]))
            else:
                ops.append(ConstantOp(IntegerAttr(random.randint(1, 1000), i32)))
        ops.append(TestOp([ops[(size // 2) * 2]]))
        return ModuleOp(ops)

    @classmethod
    def constant_folding(cls, size: int = 100) -> str:
        """Generate a constant folding workload of a given size."""
        return str(cls.constant_folding_module(size=size))

    @classmethod
    def large_dense_attr(cls, x: int = 1024, y: int = 1024) -> str:
        """Get the MLIR text representation of a large dense attr."""
        return str(cls.large_dense_attr_module(x=x, y=y))

    @classmethod
    def large_dense_attr_module(cls, x: int = 1024, y: int = 1024) -> ModuleOp:
        """Get the MLIR text representation of a large dense attr.

        An example of running
        `print(WorkloadBuilder().large_dense_attr_module(x=3, y=3))`
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
        dense_attr = [random.randint(-128, 128) for _ in range(x * y)]
        tensor_type = TensorType(element_type=i8, shape=[x, y])
        return ModuleOp(
            [
                ConstantOp(
                    DenseIntOrFPElementsAttr.from_list(
                        type=tensor_type, data=dense_attr
                    )
                )
            ]
        )

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
        # Each dense attr item is a byte = 2 hex chars
        dense_attr_hex = "".join(random.choice(HEX_CHARS) for _ in range(x * y * 2))
        ops = [
            (
                '%0 = "arith.constant"() '
                f'<{{value = dense<"0x{dense_attr_hex}"> '
                f": tensor<{x}x{y}xi8>}}> : () -> tensor<{x}x{y}xi8>"
            )
        ]
        return WorkloadBuilder.wrap_module(ops)
