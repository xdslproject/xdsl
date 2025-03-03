#!/usr/bin/env python3
"""Microbenchmark properties of the xDSL implementation."""

from __future__ import annotations

from xdsl.ir import Block
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    traits_def,
)
from xdsl.traits import OpTrait


@irdl_op_definition
class EmptyOp(IRDLOperation):
    """An empty operation."""

    name = "empty"


class TraitA(OpTrait):
    """An example trait."""


class TraitB(OpTrait):
    """An example trait."""


@irdl_op_definition
class HasTraitAOp(IRDLOperation):
    """An operation which has a trait A."""

    name = "has_trait_a"
    traits = traits_def(TraitA())


class IRTraversal:
    """Benchmark the time to traverse xDSL IR."""

    EXAMPLE_BLOCK_NUM_OPS = 1_000
    EXAMPLE_OPS = (EmptyOp() for _ in range(EXAMPLE_BLOCK_NUM_OPS))
    EXAMPLE_BLOCK = Block(ops=EXAMPLE_OPS)

    def time_iterate_ops(self) -> None:
        """Time directly iterating over a python list of operations.

        Comparison with `for (Operation *op : /*std::vector*/ops) {` at
        0.35ns/op.
        """
        for op in IRTraversal.EXAMPLE_OPS:
            assert op

    def time_iterate_block_ops(self) -> None:
        """Time directly iterating over the linked list of a block's operations.

        Comparison with `for (Operation &op : *block) {` at 2.15ns/op.
        """
        for op in IRTraversal.EXAMPLE_BLOCK.ops:
            assert op

    def time_walk_block_ops(self) -> None:
        """Time walking a block's operations.

        Comparison with `block->walk([](Operation *op) {});` with no region in
        the IR at 6.11ns/op.
        """
        for op in IRTraversal.EXAMPLE_BLOCK.walk():
            assert op


class Extensibility:
    """Benchmark the time to check interface and trait properties."""

    HAS_TRAIT_A_OP = HasTraitAOp()

    def time_interface_check(self) -> None:
        """Time checking the class hierarchy of an operation.

        Indirect comparison with `assert( dyn_cast<OpT>(op) )` at
        9.68ns/op.

        This is not a direct comparison as xDSL does not use the
        class hierarchy to express interface functionality, but is interesting
        to compare `isinstance` with `dyn_cast` in context.
        """
        assert isinstance(Extensibility.HAS_TRAIT_A_OP, HasTraitAOp)

    def time_trait_check(self) -> None:
        """Time checking the trait of an operation.

        Comparison with `assert( op->hasTrait<TraitT>(op) )` at 18.1ns/op.
        """
        assert Extensibility.HAS_TRAIT_A_OP.has_trait(TraitA)

    def time_trait_check_neg(self) -> None:
        """Time checking the trait of an operation.

        Comparison with `assert( ! op->hasTrait<TraitT>(op) )` at 13.4ns/op.
        """
        assert not Extensibility.HAS_TRAIT_A_OP.has_trait(TraitB)


class OpCreation:
    """Benchmark creating an operation in xDSL."""

    CONSTANT_OPERATION = EmptyOp()

    def time_operation_create(self) -> None:
        """Time creating an empty operation.

        Comparison with `OperationState opState(unknownLoc, "testbench.empty");
        Operation::create(opState)` at 118ns/op.
        """
        EmptyOp()

    def time_operation_clone(self) -> None:
        """Time cloning an empty operation.

        Comparison with `OwningOpRef<ModuleOp> moduleClone = moduleOp->clone();`
        at 631ns/op.
        """
        OpCreation.CONSTANT_OPERATION.clone()


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    EXTENSIBILITY = Extensibility()
    IR_TRAVERSAL = IRTraversal()
    OP_CREATION = OpCreation()

    BENCHMARKS: dict[str, Callable[[], None]] = {
        "IRTraversal.iterate_ops": IR_TRAVERSAL.time_iterate_ops,
        "IRTraversal.iterate_block_ops": IR_TRAVERSAL.time_iterate_block_ops,
        "IRTraversal.walk_block_ops": IR_TRAVERSAL.time_walk_block_ops,
        "Extensibility.interface_check": EXTENSIBILITY.time_interface_check,
        "Extensibility.trait_check": EXTENSIBILITY.time_trait_check,
        "Extensibility.trait_check_neg": EXTENSIBILITY.time_trait_check_neg,
        "OpCreation.operation_create": OP_CREATION.time_operation_create,
        "OpCreation.operation_clone": OP_CREATION.time_operation_clone,
    }
    profile(BENCHMARKS)
