#!/usr/bin/env python3
"""Workloads for benchmarking xDSL."""

import random
from math import prod

from xdsl.dialects.arith import AddfOp, AddiOp, ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FunctionType,
    IntegerAttr,
    ModuleOp,
    TensorType,
    f32,
    i8,
    i32,
)
from xdsl.dialects.func import FuncOp, ReturnOp
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

    @classmethod
    def large_constant_tensor(
        cls,
        tensor_shape: tuple[int, ...],
        num_add_ops: int = 10,
    ) -> ModuleOp:
        """Create a module with a function that adds multiple large constant tensors.

        Args:
            tensor_shape: The shape of the constant tensors to be created.
            num_add_ops: The number of addition operations to perform with the constant tensors.

        Returns:
            The created module containing the function with addition operations.
        """
        assert num_add_ops >= 0
        random.seed(RANDOM_SEED)
        tensor_type = TensorType(shape=tensor_shape, element_type=f32)
        function_type = FunctionType.from_lists(
            inputs=[tensor_type], outputs=[tensor_type]
        )

        func_op = FuncOp(name="main", function_type=function_type)

        input_val = func_op.args[0]
        output_val = input_val

        for _ in range(num_add_ops):
            # Create a constant tensor with random values
            dense_elements_attr = DenseIntOrFPElementsAttr.from_list(
                type=tensor_type,
                data=[random.random() for _ in range(prod(tensor_shape))],
            )
            const_op = ConstantOp(value=dense_elements_attr, value_type=tensor_type)
            func_op.body.block.add_op(const_op)

            # Add the constant to the input value
            add_op = AddfOp(
                operand1=input_val, operand2=const_op, result_type=tensor_type
            )
            func_op.body.block.add_op(add_op)

            # Update input_val to the result of the addition
            input_val = add_op.result
            output_val = add_op.result

        return_op = ReturnOp(output_val)
        func_op.body.block.add_op(return_op)

        module_op = ModuleOp([func_op])
        module_op.verify()
        return module_op
