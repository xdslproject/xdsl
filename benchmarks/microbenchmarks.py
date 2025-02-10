#!/usr/bin/env python3
"""Microbenchmark properties of the xDSL implementation."""

from __future__ import annotations
import importlib

from xdsl.traits import IsTerminator, NoTerminator
from xdsl.dialects.gpu import TerminatorOp
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_successor_def,
    traits_def,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Block
import xdsl.dialects.arith
import xdsl.dialects.builtin

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, Float64Type, TensorType, f64
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class IsTerminatorOp(IRDLOperation):
    """An operation that provides the IsTerminator trait."""

    name = "test.is_terminator"

    successor = opt_successor_def()

    traits = traits_def(IsTerminator())


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


class Extensibility:
    """Benchmark the time to check interface and trait properties."""

    IS_TERMINATOR_OP = TerminatorOp()

    def time_interface_check(self) -> None:
        """Time checking the class hierarchy of an operation."""
        assert isinstance(Extensibility.IS_TERMINATOR_OP, TerminatorOp)

    def time_trait_check(self) -> None:
        """Time checking the trait of an operation."""
        assert Extensibility.IS_TERMINATOR_OP.has_trait(IsTerminator)
        assert not Extensibility.IS_TERMINATOR_OP.has_trait(NoTerminator)


class ImportClasses:
    """Benchmark the time to import xDSL classes."""

    def time_import_xdsl_opt(self) -> None:
        """Import benchmark using the default asv mechanism."""
        from xdsl.xdsl_opt_main import xDSLOptMain  # noqa: F401


    def timeraw_import_xdsl_opt(self) -> str:
        """Import benchmark using the `raw` asv mechanism."""
        return """
        from xdsl.xdsl_opt_main import xDSLOptMain
        """

class IRTraversal:
    """Benchmark the time to traverse xDSL IR."""

    EXAMPLE_BLOCK_NUM_OPS = 1_000
    EXAMPLE_BLOCK = Block(ops=(TestOp() for _ in range(EXAMPLE_BLOCK_NUM_OPS)))

    def time_iterate_block_ops(self) -> None:
        """Time directly iterating over a block's operations."""
        for op in IRTraversal.EXAMPLE_BLOCK.ops:
            assert op


    def time_walk_block_ops(self) -> None:
        """Time walking a block's operations."""
        for op in IRTraversal.EXAMPLE_BLOCK.walk():
            assert op


class LoadDialects:
    """Benchmark loading dialects in xDSL."""

    def time_arith_load(self) -> None:
        """Time loading the `arith` dialect.

        Note that this must be done with `importlib.reload` rather than just
        directly importing with `from xdsl.dialects.arith import Arith` to avoid
        tests interacting with each other.
        """
        importlib.reload(xdsl.dialects.arith)


    def time_builtin_load(self) -> None:
        """Time loading the `builtin` dialect."""
        importlib.reload(xdsl.dialects.builtin)


class OpCreation:
    """Benchmark creating an operation in xDSL."""

    CONSTANT_OPERATION_X_SIZE = 6
    CONSTANT_OPERATION_Y_SIZE = 6
    CONSTANT_OPERATION = ConstantOp.from_list(
        [x for x in range(CONSTANT_OPERATION_X_SIZE)],
        [y for y in range(CONSTANT_OPERATION_Y_SIZE)],
    )


    def time_operation_create(self) -> None:
        """Time creating an operation."""
        ConstantOp.from_list(
            [x for x in range(OpCreation.CONSTANT_OPERATION_X_SIZE)],
            [y for y in range(OpCreation.CONSTANT_OPERATION_Y_SIZE)],
        )


    def time_operation_clone(self) -> None:
        """Time cloning an operation."""
        OpCreation.CONSTANT_OPERATION.clone()


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    EXTENSIBILITY = Extensibility()
    IMPORT_CLASSES = ImportClasses()
    IR_TRAVERSAL = IRTraversal()
    LOAD_DIALECTS = LoadDialects()
    OP_CREATION = OpCreation()

    BENCHMARKS: dict[str, Callable[[], None]] = {
       "Extensibility.interface_check": EXTENSIBILITY.time_interface_check,
       "Extensibility.trait_check": EXTENSIBILITY.time_trait_check,
       "ImportClasses.import_xdsl_opt": IMPORT_CLASSES.time_import_xdsl_opt,
       "IRTraversal.time_iterate_block_ops": IR_TRAVERSAL.time_iterate_block_ops,
       "IRTraversal.time_walk_block_ops": IR_TRAVERSAL.time_walk_block_ops,
       "LoadDialects.time_arith_load": LOAD_DIALECTS.time_arith_load,
       "LoadDialects.time_builtin_load": LOAD_DIALECTS.time_builtin_load,
       "OpCreation.time_operation_create": OP_CREATION.time_operation_create,
       "OpCreation.time_operation_clone": OP_CREATION.time_operation_clone,
    }
    profile(BENCHMARKS)
