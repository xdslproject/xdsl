#!/usr/bin/env python3
"""Benchmarks for the pattern rewriter of the xDSL implementation."""

from benchmarks.helpers import get_context, parse_module
from benchmarks.workloads import WorkloadBuilder
from xdsl.dialects.arith import AddiOp, Arith, ConstantOp, SubiOp
from xdsl.dialects.builtin import Builtin, IntegerAttr, ModuleOp
from xdsl.pattern_rewriter import Worklist
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import is_trivially_dead

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


class RewritingMicrobenchmarks:
    """Microbenchmark rewriting in xDSL."""

    WORKLOAD_CONSTANT_20 = parse_module(CTX, WorkloadBuilder.constant_folding(20))
    workload_constant_20: ModuleOp
    const_0: ConstantOp
    const_1: ConstantOp
    add_op: AddiOp
    sub_op: SubiOp

    def setup(self) -> None:
        """Setup the benchmarks."""
        self.setup_operations()
        self.setup_constant_folding_20()

    def setup_operations(self) -> None:
        """Setup example operations"""
        self.const_0 = ConstantOp(IntegerAttr(0, 64))
        self.const_1 = ConstantOp(IntegerAttr(1, 64))
        self.add_op = AddiOp(self.const_1, self.const_0)
        self.sub_op = SubiOp(self.const_1, self.const_0)

    def setup_constant_folding_20(self) -> None:
        """Setup the constant folding 20 items benchmark."""
        self.workload_constant_20 = PatternRewriter.WORKLOAD_CONSTANT_20.clone()

    def time_get_variadic_sizes(self) -> None:
        """Time getting the variadic size of an operation."""

    def time_region_walk(self) -> None:
        """Time walking a region."""
        for block in self.workload_constant_20.body.walk():
            assert block

    def setup_worklist(self) -> None:
        """Setup a worklist"""
        self.worklist = Worklist()
        self.worklist.push(self.sub_op)

    def time_worklist_push(self) -> None:
        """Time pushing to a worklist."""
        self.worklist.push(self.add_op)

    def time_worklist_pop(self) -> None:
        """Time popping from a worklist."""
        self.worklist.pop()

    def time_insert_point_before(self) -> None:
        """Time getting the insertion point before an operation."""

    def time_is_trivially_dead(self) -> None:
        """Time checking if an operation is trivially dead."""
        is_trivially_dead(self.add_op)

    def time_get_trait(self) -> None:
        """Time getting a trait from an operation."""
        self.add_op.get_trait(HasCanonicalizationPatternsTrait)

    def time_const_evaluate_operand(self) -> None:
        """Time trying constant evaluate an SSA value."""

    def time_integer_type_normalized_value(self) -> None:
        """Time `IntegerType.normalized_value`."""

    def time_parameterised_attribute_init(self) -> None:
        """Time ."""

    def time_pattern_rewriter_insert_op(self) -> None:
        """Time `PatternRewriter.insert_op`."""

    def time_ssa_value_replace_by(self) -> None:
        """Time `SSAValue.replace_by`."""

    def time_patter_rewriter_erase_op(self) -> None:
        """Time `PatternRewriter.erase_op`."""

    def time_result_only_effects(self) -> None:
        """Time `result_only_effects`."""

    def time_liveset_propagate_region_liveness(self) -> None:
        """Time `LiveSet.propagate_region_liveness`."""

    def time_liveset_propagate_op_liveness(self) -> None:
        """Time `LiveSet.propagate_op_liveness`."""

    def time_liveset_delete_dead(self) -> None:
        """Time `LiveSet.delete_dead`."""

    # TODO: milli-benchmark of top-level rewrites
    # TODO: extend_from_listener
    # TODO: `result_only_effects` > `get_effects` or `Operation.get_traits_of_type`
    # def time_(self) -> None:
    #     """Time ."""


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    PATTERN_REWRITER = PatternRewriter()
    REWRITER_UBENCHMARKS = RewritingMicrobenchmarks()
    profile(
        {
            "PatternRewriter.constant_folding_20": Benchmark(
                PATTERN_REWRITER.time_constant_folding_20,
                PATTERN_REWRITER.setup_constant_folding_20,
            ),
            "PatternRewriter.constant_folding_100": Benchmark(
                PATTERN_REWRITER.time_constant_folding_100,
                PATTERN_REWRITER.setup_constant_folding_100,
            ),
            "PatternRewriter.constant_folding_1000": Benchmark(
                PATTERN_REWRITER.time_constant_folding_1000,
                PATTERN_REWRITER.setup_constant_folding_1000,
            ),
        }
    )
