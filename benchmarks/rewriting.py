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
from xdsl.ir import ParametrizedAttribute, Region
from xdsl.irdl import OpDef, VarIRConstruct, get_variadic_sizes, traits_def
from xdsl.pattern_rewriter import Worklist
from xdsl.rewriter import InsertPoint
from xdsl.traits import Commutative, HasCanonicalizationPatternsTrait, Pure
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
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
        # Operations
        ops = self.region.walk()
        self.const_0 = cast(ConstantOp, next(ops))
        self.const_1 = cast(ConstantOp, next(ops))
        self.add_op = cast(AddiOp, next(ops))
        self.add_op_def = OpDef.from_pyrdl(AddiOpUnwrapped)
        self.add_op_construct = VarIRConstruct.OPERAND
        self.sub_op = SubiOp(self.const_1, self.const_0)
        # Worklist
        self.worklist = Worklist()
        self.worklist.push(self.sub_op)
        # Types
        self.integer_type = IntegerType(64)
        # LiveSet

    # ======================================================================== #

    def time_get_variadic_sizes(self) -> None:
        """Time getting the variadic size of an operation."""
        get_variadic_sizes(self.add_op, self.add_op_def, self.add_op_construct)

    def time_region_walk(self) -> None:
        """Time walking a region."""
        for block in self.region.walk():
            assert block

    def time_worklist_push(self) -> None:
        """Time pushing to a worklist."""
        self.worklist.push(self.add_op)

    def time_worklist_pop(self) -> None:
        """Time popping from a worklist."""
        self.worklist.pop()

    def time_is_trivially_dead(self) -> None:
        """Time checking if an operation is trivially dead."""
        is_trivially_dead(self.add_op)

    def time_get_trait(self) -> None:
        """Time getting a trait from an operation."""
        self.add_op.get_trait(HasCanonicalizationPatternsTrait)

    def time_const_evaluate_operand(self) -> None:
        """Time trying constant evaluate an SSA value."""
        const_evaluate_operand(self.add_op.lhs)

    def time_integer_type_normalized_value(self) -> None:
        """Time `IntegerType.normalized_value`."""
        self.integer_type.normalized_value(0)

    def time_parameterised_attribute_init(self) -> None:
        """Time instantiating a parameterised attribute."""
        ParametrizedAttribute(parameters=[IntegerAttr(0, 64), IntegerAttr(0, 64)])

    def time_operation_create(self) -> None:
        """Time creating an operation."""
        AddiOp(self.const_1, self.const_0)

    def time_insert_point_before(self) -> None:
        """Time getting the insertion point before an operation."""
        InsertPoint.before(self.add_op)

    # def time_pattern_rewriter_insert_op(self) -> None:
    #     """Time `PatternRewriter.insert_op`."""
    #     # `insertion_point.block.insert_ops_before(ops, insertion_point.insert_before)`
    #     raise NotImplementedError()

    def time_block_detach_op(self) -> None:
        """Time detaching an operation from a block."""
        self.region.block.detach_op(self.add_op)

    def time_operation_drop_all_references(self) -> None:
        """Time dropping all references to an operation."""
        self.add_op.drop_all_references()

    def time_ssavalue_erase(self) -> None:
        """Time erasing an SSA value."""
        self.add_op.result.erase()

    def time_result_only_effects(self) -> None:
        """Time `result_only_effects`."""
        raise NotImplementedError()

    def time_set_live(self) -> None:
        """Time `LiveSet.set_live`."""
        raise NotImplementedError()

    def time_liveset_delete_dead(self) -> None:
        """Time `LiveSet.delete_dead`."""
        raise NotImplementedError()

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
                PATTERN_REWRITER.setup,
            ),
            # "PatternRewriter.constant_folding_100": Benchmark(
            #     PATTERN_REWRITER.time_constant_folding_100,
            #     PATTERN_REWRITER.setup_constant_folding_100,
            # ),
            # "PatternRewriter.constant_folding_1000": Benchmark(
            #     PATTERN_REWRITER.time_constant_folding_1000,
            #     PATTERN_REWRITER.setup_constant_folding_1000,
            # ),
            # ================================================================ #
            "RewritingMicrobenchmarks.get_variadic_sizes": Benchmark(
                REWRITER_UBENCHMARKS.time_get_variadic_sizes, REWRITER_UBENCHMARKS.setup
            ),
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
            "RewritingMicrobenchmarks.is_trivially_dead": Benchmark(
                REWRITER_UBENCHMARKS.time_is_trivially_dead,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.get_trait": Benchmark(
                REWRITER_UBENCHMARKS.time_get_trait,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.const_evaluate_operand": Benchmark(
                REWRITER_UBENCHMARKS.time_const_evaluate_operand,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.integer_type_normalized_value": Benchmark(
                REWRITER_UBENCHMARKS.time_integer_type_normalized_value,
                REWRITER_UBENCHMARKS.setup,
            ),
            # "RewritingMicrobenchmarks.parameterised_attribute_init": Benchmark(
            #     REWRITER_UBENCHMARKS.time_parameterised_attribute_init,
            #     REWRITER_UBENCHMARKS.setup,
            # ),
            "RewritingMicrobenchmarks.operation_create": Benchmark(
                REWRITER_UBENCHMARKS.time_operation_create,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.insert_point_before": Benchmark(
                REWRITER_UBENCHMARKS.time_insert_point_before,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.block_detach_op": Benchmark(
                REWRITER_UBENCHMARKS.time_block_detach_op,
                REWRITER_UBENCHMARKS.setup,
            ),
            "RewritingMicrobenchmarks.operation_drop_all_references": Benchmark(
                REWRITER_UBENCHMARKS.time_operation_drop_all_references,
                REWRITER_UBENCHMARKS.setup,
            ),
            # "RewritingMicrobenchmarks.ssavalue_erase": Benchmark(
            #     REWRITER_UBENCHMARKS.time_ssavalue_erase,
            #     REWRITER_UBENCHMARKS.setup,
            # ),
            # "RewritingMicrobenchmarks.result_only_effects": Benchmark(
            #     REWRITER_UBENCHMARKS.time_result_only_effects,
            #     REWRITER_UBENCHMARKS.setup,
            # ),
            # "RewritingMicrobenchmarks.set_live": Benchmark(
            #     REWRITER_UBENCHMARKS.time_set_live,
            #     REWRITER_UBENCHMARKS.setup,
            # ),
            # "RewritingMicrobenchmarks.liveset_delete_dead": Benchmark(
            #     REWRITER_UBENCHMARKS.time_liveset_delete_dead,
            #     REWRITER_UBENCHMARKS.setup,
            # ),
        }
    )
