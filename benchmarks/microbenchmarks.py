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


NUM_CONSTRUCTED_TRAITS = 8


def get_optrait_subclass() -> type[OpTrait]:
    """Construct a unique subclass of `OpTrait`."""

    class Trait(OpTrait):
        pass

    return Trait


optrait_subclasses: dict[int, type[OpTrait]] = {
    i + 1: get_optrait_subclass() for i in range(NUM_CONSTRUCTED_TRAITS)
}


@irdl_op_definition
class HasManyTraitOp(IRDLOperation):
    """An operation which has many traits."""

    name = "has_trait_a"
    traits = traits_def(*[trait() for trait in optrait_subclasses.values()])


class IRTraversal:
    """Benchmark the time to traverse xDSL IR."""

    EXAMPLE_BLOCK_NUM_OPS = 1_000
    EXAMPLE_OPS = (EmptyOp() for _ in range(EXAMPLE_BLOCK_NUM_OPS))
    EXAMPLE_BLOCK = Block(ops=EXAMPLE_OPS)

    def time_iterate_ops(self) -> None:
        """Time directly iterating over a python list of operations.

        For comparison with the "How Slow is MLIR" testbench
        `IRWalk/vectorTraveral`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        SmallVector<Operation *> ops;
        for (int j = 0; j < state.range(0); ++j) {
            ops.push_back(b.create<EmptyOp>(unknownLoc));
        }
        for (auto _ : state) {
            for (Operation *op : ops) {
                benchmark::DoNotOptimize(op);
            };
        }
        ```
        """
        for op in IRTraversal.EXAMPLE_OPS:
            op  # pyright: ignore[reportUnusedExpression]

    def time_iterate_block_ops(self) -> None:
        """Time directly iterating over the linked list of a block's operations.

        For comparison with the "How Slow is MLIR" testbench
        `IRWalk/blockTraveral`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        for (int j = 0; j < state.range(0); ++j) {
            b.create<EmptyOp>(unknownLoc);
        }
        Block *block = moduleOp->getBody();
        for (auto _ : state) {
            for (Operation &op : *block) {
                benchmark::DoNotOptimize(&op);
            };
        }
        ```
        """
        for op in IRTraversal.EXAMPLE_BLOCK.ops:
            op  # pyright: ignore[reportUnusedExpression]

    def time_walk_block_ops(self) -> None:
        """Time walking a block's operations.

        For comparison with the "How Slow is MLIR" testbench
        `IRWalk/blockTraveral`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        for (int j = 0; j < state.range(0); ++j) {
            b.create<EmptyOp>(unknownLoc);
        }
        Block *block = moduleOp->getBody();
        for (auto _ : state) {
            block->walk([](Operation *op) { benchmark::DoNotOptimize(&op); });
        }
        ```
        """
        for op in IRTraversal.EXAMPLE_BLOCK.walk():
            op  # pyright: ignore[reportUnusedExpression]


class Extensibility:
    """Benchmark the time to check interface and trait properties."""

    EMPTY_OP = EmptyOp()
    HAS_TRAIT_A_OP = HasTraitAOp()
    OP_WITH_REGION = HasManyTraitOp()
    TRAIT_4 = optrait_subclasses[4]
    TRAIT_4_INSTANCE = TRAIT_4()

    def time_interface_check_trait(self) -> None:
        """Time checking the class hierarchy of a trait."""
        isinstance(Extensibility.HAS_TRAIT_A_OP, HasTraitAOp)  # pyright: ignore[reportUnnecessaryIsInstance]

    def time_interface_check(self) -> None:
        """Time checking the class hierarchy of an operation.

        For comparison with the "How Slow is MLIR" testbench
        `IRWalk/vectorTraveralOpCastSuccess`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        SmallVector<Operation *> ops;
        for (int j = 0; j < state.range(0); ++j) {
            ops.push_back(b.create<EmptyOp>(unknownLoc));
        }
        for (auto _ : state) {
            for (Operation *op : ops) {
                auto casted = dyn_cast<EmptyOp>(op);
                benchmark::DoNotOptimize(&casted);
            };
        }
        ```
        """
        isinstance(Extensibility.EMPTY_OP, EmptyOp)  # pyright: ignore[reportUnnecessaryIsInstance]

    def time_trait_check(self) -> None:
        """Time checking the trait of an operation.

        For comparison with the "How Slow is MLIR" testbench
        `IRWalk/vectorTraveralOpTraitSuccess`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        SmallVector<Operation *> ops;
        for (int j = 0; j < state.range(0); ++j) {
            ops.push_back(b.create<OpWithRegion>(unknownLoc));
        }
        for (auto _ : state) {
            for (Operation *op : ops) {
                bool hasTrait = op->hasTrait<OpTrait::SingleBlock>();
                benchmark::DoNotOptimize(&hasTrait);
            };
        }
        ```

        Since MLIR provides the following traits `mlir::OpTrait::OneRegion`,
        `mlir::OpTrait::ZeroResults`, `mlir::OpTrait::ZeroSuccessors`,
        `mlir::OpTrait::ZeroOperands`, `mlir::OpTrait::SingleBlock`,
        `mlir::OpTrait::OpInvariants`, `mlir::RegionKindInterface::Trait`,
        `mlir::OpTrait::HasOnlyGraphRegion`, our constructed operation also
        has eight traits for fair comparison.
        """
        assert Extensibility.OP_WITH_REGION.has_trait(Extensibility.TRAIT_4)

    def time_trait_check_optimised(self) -> None:
        """Time checking the trait of an operation using optimised code."""
        has_trait = False
        for t in Extensibility.OP_WITH_REGION.traits._traits:  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues, reportPrivateUsage]
            if isinstance(t, Extensibility.TRAIT_4):
                has_trait = True
                break
        assert has_trait

    def time_trait_check_single(self) -> None:
        """Time checking the trait of an operation with one trait."""
        Extensibility.HAS_TRAIT_A_OP.has_trait(TraitA)

    def time_trait_check_neg(self) -> None:
        """Time checking the trait of an operation.

        For comparison with the "How Slow is MLIR" testbench
        `IRWalk/vectorTraveralOpTraitFail`, implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        SmallVector<Operation *> ops;
        for (int j = 0; j < state.range(0); ++j) {
            ops.push_back(b.create<EmptyOp>(unknownLoc));
        }
        for (auto _ : state) {
            for (Operation *op : ops) {
                bool hasTrait = op->hasTrait<OpTrait::SingleBlock>();
                benchmark::DoNotOptimize(&hasTrait);
            };
        }
        ```
        """
        Extensibility.OP_WITH_REGION.has_trait(TraitB)


class OpCreation:
    """Benchmark creating an operation in xDSL."""

    CONSTANT_OPERATION = EmptyOp()

    def time_operation_create(self) -> None:
        """Time creating an empty operation.

        For comparison with the "How Slow is MLIR" testbench
        `CreateOps/hoistedOpState`, implemented as:

        ```
        OperationState opState(unknownLoc, "testbench.empty");
        for (auto _ : state) {
            for (int j = 0; j < state.range(0); ++j)
                Operation::create(opState);
        }
        ```
        """
        EmptyOp.create()

    def time_operation_build(self) -> None:
        """Time building an empty operation.

        For comparison with the "How Slow is MLIR" testbench
        `CreateOps/llvm_withInsertRegistered`, implemented as:

        ```
        auto module = std::make_unique<llvm::Module>("MyModule", ctx);
        auto *fTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
        auto *func = llvm::Function::Create(fTy, llvm::Function::ExternalLinkage,
                                            "", module.get());
        auto *block = llvm::BasicBlock::Create(ctx, "", func);
        llvm::IRBuilder<> builder(block);
        for (auto _ : state) {
            for (int j = 0; j < state.range(0); ++j)
                builder.CreateUnreachable();
        }
        ```
        """
        EmptyOp.build()

    def time_operation_clone(self) -> None:
        """Time cloning an empty operation.

        For comparison with the "How Slow is MLIR" testbench `Cloning/cloneOps`,
        implemented as:

        ```
        ctx->loadDialect<TestBenchDialect>();
        OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
        // Create a bunch of ops that have operands.
        Value operand = moduleOp->getBody()->addArgument(b.getI32Type(), unknownLoc);
        for (int i = 0; i < state.range(0); ++i) {
            operand = b.create<PassthroughOp>(unknownLoc, b.getI32Type(), operand);
        }
        for (auto _ : state) {
            OwningOpRef<ModuleOp> moduleClone = moduleOp->clone();
            benchmark::DoNotOptimize(moduleClone.get());
        }
        ```
        """
        OpCreation.CONSTANT_OPERATION.clone()


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    EXTENSIBILITY = Extensibility()
    IR_TRAVERSAL = IRTraversal()
    OP_CREATION = OpCreation()
    profile(
        {
            "IRTraversal.iterate_ops": Benchmark(IR_TRAVERSAL.time_iterate_ops),
            "IRTraversal.iterate_block_ops": Benchmark(
                IR_TRAVERSAL.time_iterate_block_ops
            ),
            "IRTraversal.walk_block_ops": Benchmark(IR_TRAVERSAL.time_walk_block_ops),
            "Extensibility.interface_check_trait": Benchmark(
                EXTENSIBILITY.time_interface_check_trait
            ),
            "Extensibility.interface_check": Benchmark(
                EXTENSIBILITY.time_interface_check
            ),
            "Extensibility.trait_check": Benchmark(EXTENSIBILITY.time_trait_check),
            "Extensibility.trait_check_optimised": Benchmark(
                EXTENSIBILITY.time_trait_check_optimised
            ),
            "Extensibility.trait_check_single": Benchmark(
                EXTENSIBILITY.time_trait_check_single
            ),
            "Extensibility.trait_check_neg": Benchmark(
                EXTENSIBILITY.time_trait_check_neg
            ),
            "OpCreation.operation_create": Benchmark(OP_CREATION.time_operation_create),
            "OpCreation.operation_build": Benchmark(OP_CREATION.time_operation_build),
            "OpCreation.operation_clone": Benchmark(OP_CREATION.time_operation_clone),
        }
    )
