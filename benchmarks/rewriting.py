#!/usr/bin/env python3
"""Benchmarks for the pattern rewriter of the xDSL implementation."""

from typing import cast

from benchmarks.helpers import get_context, parse_module
from benchmarks.workloads import WorkloadBuilder
from xdsl.dialects.arith import (
    AddiOp,
    Arith,
    ConstantOp,
    SignlessIntegerBinaryOperationHasCanonicalizationPatternsTrait,
    SignlessIntegerBinaryOperationWithOverflow,
    SubiOp,
)
from xdsl.dialects.builtin import Builtin, IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import Region
from xdsl.ir.post_order import PostOrderIterator
from xdsl.irdl import OpDef, VarIRConstruct, get_variadic_sizes, traits_def
from xdsl.pattern_rewriter import Worklist
from xdsl.rewriter import InsertPoint
from xdsl.traits import (
    Commutative,
    HasCanonicalizationPatternsTrait,
    MemoryEffect,
    Pure,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import (
    LiveSet,
    is_trivially_dead,
    result_only_effects,
    would_be_trivially_dead,
)

CTX = get_context()
CTX.load_dialect(Arith)
CTX.load_dialect(Builtin)

CANONICALIZE_PASS = CanonicalizePass()


class PatternRewriter:
    """Benchmark rewriting in xDSL."""

    WORKLOAD_CONSTANT_20 = parse_module(CTX, WorkloadBuilder.constant_folding(20))
    WORKLOAD_CONSTANT_100 = parse_module(CTX, WorkloadBuilder.constant_folding(100))
    WORKLOAD_CONSTANT_1000 = parse_module(CTX, WorkloadBuilder.constant_folding(1_000))

    workload_constant_20: ModuleOp
    workload_constant_100: ModuleOp
    workload_constant_1000: ModuleOp

    def setup(self) -> None:
        """Setup the benchmarks."""
        self.setup_constant_folding_20()
        self.setup_constant_folding_100()
        self.setup_constant_folding_1000()

    def setup_constant_folding_20(self) -> None:
        """Setup the constant folding 20 items benchmark."""
        self.workload_constant_20 = PatternRewriter.WORKLOAD_CONSTANT_20.clone()

    def time_constant_folding_20(self) -> None:
        """Time canonicalizing constant folding for 20 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_20)

    def setup_constant_folding_100(self) -> None:
        """Setup the constant folding 100 items benchmark."""
        self.workload_constant_100 = PatternRewriter.WORKLOAD_CONSTANT_100.clone()

    def time_constant_folding_100(self) -> None:
        """Time canonicalizing constant folding for 100 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_100)

    def setup_constant_folding_1000(self) -> None:
        """Setup the constant folding 1000 items benchmark."""
        self.workload_constant_1000 = PatternRewriter.WORKLOAD_CONSTANT_1000.clone()

    def time_constant_folding_1000(self) -> None:
        """Time canonicalizing constant folding for 1000 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_1000)


class AddiOpUnwrapped(SignlessIntegerBinaryOperationWithOverflow):
    """Unwrapped version of `AddiOp` to allow `OpDef.from_pyrdl(...)`."""

    name = "arith.addi"

    traits = traits_def(
        Pure(),
        Commutative(),
        SignlessIntegerBinaryOperationHasCanonicalizationPatternsTrait(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int | None:
        return lhs + rhs

    @staticmethod
    def is_right_unit(attr: IntegerAttr) -> bool:
        return attr.value.data == 0


class RewritingMicrobenchmarks:
    """Microbenchmark rewriting in xDSL."""

    WORKLOAD_CONSTANT_20 = parse_module(CTX, WorkloadBuilder.constant_folding(20))
    region: Region
    const_0: ConstantOp
    const_1: ConstantOp
    add_op: AddiOp
    sub_op: SubiOp
    integer_type: IntegerType
    # attribute_sequence: Sequence[Attribute]

    def setup(self) -> None:
        """Setup the benchmarks."""
        # Region
        self.region = PatternRewriter.WORKLOAD_CONSTANT_20.clone().body
        assert self.region._first_block is not None
        self.first_block = self.region._first_block
        # Operations
        ops = self.region.walk()
        self.const_0 = cast(ConstantOp, next(ops))
        self.const_1 = cast(ConstantOp, next(ops))
        self.add_op = cast(AddiOp, next(ops))
        self.add_op_def = OpDef.from_pyrdl(AddiOpUnwrapped)
        self.add_op_construct = VarIRConstruct.OPERAND
        self.add_op_result = self.add_op.result
        self.add_op_result_use = list(self.add_op_result.uses)[0]
        self.sub_op = SubiOp(self.const_1, self.const_0)
        self.sub_op_result = self.sub_op.result
        # Worklist
        self.worklist = Worklist()
        self.worklist.push(self.sub_op)
        # Types
        self.integer_type = IntegerType(64)
        self.integer_attr = IntegerAttr(0, 64)
        # LiveSet
        self.live_set = LiveSet()

    # =================== #
    # Worklist operations #
    # =================== #

    def time_region_walk(self) -> None:
        """Time `Region.walk`."""
        for block in self.region.walk():
            assert block

    def time_worklist_push(self) -> None:
        """Time `Worklist.push`."""
        self.worklist.push(self.add_op)

    def time_worklist_pop(self) -> None:
        """Time `Worklist.pop`."""
        self.worklist.pop()

    # TODO: Split into separate classes
    # ================================================== #
    # `CanonicalizationRewritePattern.match_and_rewrite` #
    # ================================================== #

    def time_get_trait(self) -> None:
        """Time `Operation.get_trait`."""
        self.add_op.get_trait(HasCanonicalizationPatternsTrait)

    # ===== #

    def time_insert_point_before(self) -> None:
        """Time `InsertPoint.before`."""
        InsertPoint.before(self.add_op)

    def time_pattern_rewriter_insert_op(self) -> None:
        """Time `PatternRewriter.insert_op`."""
        # `insertion_point.block.insert_ops_before(ops, insertion_point.insert_before)`
        raise NotImplementedError()

    def time_operation_drop_all_references(self) -> None:
        """Time `Operation.drop_all_references`."""
        self.add_op.drop_all_references()

    def time_ssavalue_replace_by(self) -> None:
        """Time `SSAValue.replace_by`."""
        self.add_op_result.replace_by(self.sub_op_result)

    def time_irwithuses_remove_use(self) -> None:
        """Time `IRWithUses.remove_use`."""
        self.add_op_result.remove_use(self.add_op_result_use)

    def time_irwithuses_add_use(self) -> None:
        """Time `IRWithUses.add_use`."""
        self.sub_op_result.add_use(self.add_op_result_use)

    def time_ssavalue_name_hint(self) -> None:
        """Time `SSAValue.namehint`."""
        self.add_op_result.name_hint = "valid_name"

    def time_handle_operation_removal(self) -> None:
        """Time `PatternRewriteWalker._handle_operation_removal`."""
        raise NotImplementedError()

    def time_block_detach_op(self) -> None:
        """Time `Block.detach_op`."""
        raise NotImplementedError()
        # self.region.block.detach_op(self.add_op)

    def time_ssavalue_erase(self) -> None:
        """Time `SSAValue.erase()`."""
        raise NotImplementedError()
        # self.add_op_result.erase()

    # ===== #

    def time_get_variadic_sizes(self) -> None:
        """Time `get_variadic_sizes`."""
        get_variadic_sizes(self.add_op, self.add_op_def, self.add_op_construct)

    # ===== #

    def time_integer_attr_creation(self) -> None:
        """Time `IntegerAttr.__init__`."""
        IntegerAttr(0, 64)

    def time_integer_type_normalized_value(self) -> None:
        """Time `IntegerType.normalized_value`."""
        self.integer_type.normalized_value(0)

    def time_integer_attr_verify(self) -> None:
        """Time `IntegerAttr._verify`."""
        self.integer_attr._verify()

    def time_operation_create(self) -> None:
        """Time `AddiOp.__init__`."""
        AddiOp(self.const_1, self.const_0)

    # ===== #

    def time_const_evaluate_operand(self) -> None:
        """Time `const_evaluate_operand`."""
        const_evaluate_operand(self.add_op.lhs)

    # ========================================== #
    # `RemoveUnusedOperations.match_and_rewrite` #
    # ========================================== #

    def time_is_trivially_dead(self) -> None:
        """Time `is_trivially_dead`."""
        is_trivially_dead(self.add_op)

    def time_would_be_trivially_dead(self) -> None:
        """Time `would_be_trivially_dead`."""
        would_be_trivially_dead(self.add_op)

    def time_result_only_effects(self) -> None:
        """Time `result_only_effects`."""
        result_only_effects(self.add_op)

    def time_operation_get_traits_of_type(self) -> None:
        """Time `Operation.get_traits_of_type`."""
        self.add_op.get_traits_of_type(MemoryEffect)

    # ============ #
    # `region_dce` #
    # ============ #

    def time_post_order_iterator(self) -> None:
        """Time `PostOrderIterator`."""
        PostOrderIterator(self.first_block)

    def time_liveset_set_live(self) -> None:
        """Time `LiveSet.set_live`."""
        self.live_set.set_live(self.add_op)

    def time_liveset_delete_dead(self) -> None:
        """Time `LiveSet.delete_dead`."""
        self.live_set.delete_dead(self.region, None)


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    PATTERN_REWRITER = PatternRewriter()
    REWRITER_UBENCHMARKS = RewritingMicrobenchmarks()
    profile(
        {
            "PatternRewriter.constant_folding_20": Benchmark(
                PATTERN_REWRITER.time_constant_folding_20,
                PATTERN_REWRITER.setup,
            ),
            "PatternRewriter.constant_folding_100": Benchmark(
                PATTERN_REWRITER.time_constant_folding_100,
                PATTERN_REWRITER.setup_constant_folding_100,
            ),
            "PatternRewriter.constant_folding_1000": Benchmark(
                PATTERN_REWRITER.time_constant_folding_1000,
                PATTERN_REWRITER.setup_constant_folding_1000,
            ),
            # ================================================================ #
            "RewritingMicrobenchmarks.region_walk": Benchmark(
                REWRITER_UBENCHMARKS.time_region_walk, REWRITER_UBENCHMARKS.setup
            ),
            "RewritingMicrobenchmarks.worklist_push": Benchmark(
                REWRITER_UBENCHMARKS.time_worklist_push,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.worklist_pop": Benchmark(
                REWRITER_UBENCHMARKS.time_worklist_pop,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.get_trait": Benchmark(
                REWRITER_UBENCHMARKS.time_get_trait,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.insert_point_before": Benchmark(
                REWRITER_UBENCHMARKS.time_insert_point_before,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.pattern_rewriter_insert_op": Benchmark(
                REWRITER_UBENCHMARKS.time_pattern_rewriter_insert_op,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.operation_drop_all_references": Benchmark(
                REWRITER_UBENCHMARKS.time_operation_drop_all_references,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.ssavalue_replace_by": Benchmark(
                REWRITER_UBENCHMARKS.time_ssavalue_replace_by,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.irwithuses_remove_use": Benchmark(
                REWRITER_UBENCHMARKS.time_irwithuses_remove_use,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.irwithuses_add_use": Benchmark(
                REWRITER_UBENCHMARKS.time_irwithuses_add_use,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.ssavalue_name_hint": Benchmark(
                REWRITER_UBENCHMARKS.time_ssavalue_name_hint,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.handle_operation_removal": Benchmark(
                REWRITER_UBENCHMARKS.time_handle_operation_removal,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.block_detach_op": Benchmark(
                REWRITER_UBENCHMARKS.time_block_detach_op,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.ssavalue_erase": Benchmark(
                REWRITER_UBENCHMARKS.time_ssavalue_erase,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.get_variadic_sizes": Benchmark(
                REWRITER_UBENCHMARKS.time_get_variadic_sizes, REWRITER_UBENCHMARKS.setup
            ),
            "RewritingMicrobenchmarks.integer_attr_creation": Benchmark(
                REWRITER_UBENCHMARKS.time_integer_attr_creation,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.integer_type_normalized_value": Benchmark(
                REWRITER_UBENCHMARKS.time_integer_type_normalized_value,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.integer_attr_verify": Benchmark(
                REWRITER_UBENCHMARKS.time_integer_attr_verify,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.operation_create": Benchmark(
                REWRITER_UBENCHMARKS.time_operation_create,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.const_evaluate_operand": Benchmark(
                REWRITER_UBENCHMARKS.time_const_evaluate_operand,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.is_trivially_dead": Benchmark(
                REWRITER_UBENCHMARKS.time_is_trivially_dead,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.would_be_trivially_dead": Benchmark(
                REWRITER_UBENCHMARKS.time_would_be_trivially_dead,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.result_only_effects": Benchmark(
                REWRITER_UBENCHMARKS.time_result_only_effects,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.operation_get_traits_of_type": Benchmark(
                REWRITER_UBENCHMARKS.time_operation_get_traits_of_type,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.post_order_iterator": Benchmark(
                REWRITER_UBENCHMARKS.time_post_order_iterator,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.liveset_set_live": Benchmark(
                REWRITER_UBENCHMARKS.time_liveset_set_live,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.liveset_delete_dead": Benchmark(
                REWRITER_UBENCHMARKS.time_liveset_delete_dead,
                REWRITER_UBENCHMARKS.setup,
            ),
        }
    )
