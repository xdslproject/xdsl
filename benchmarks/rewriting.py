#!/usr/bin/env python3
"""Benchmarks for the pattern rewriter of the xDSL implementation."""

from typing import cast

from benchmarks.workloads import WorkloadBuilder
from xdsl.context import Context
from xdsl.dialects.arith import (
    AddiOp,
    Arith,
    ConstantOp,
    SubiOp,
)
from xdsl.dialects.builtin import Builtin, IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import Region
from xdsl.ir.post_order import PostOrderIterator
from xdsl.irdl import VarIRConstruct, get_variadic_sizes
from xdsl.parser import Parser as XdslParser
from xdsl.pattern_rewriter import PatternRewriter, Worklist
from xdsl.rewriter import InsertPoint
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    MemoryEffect,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import (
    LiveSet,
    is_trivially_dead,
    result_only_effects,
    would_be_trivially_dead,
)
from xdsl.transforms.test_constant_folding import (
    TestConstantFoldingPass,
    TestSpecialisedConstantFoldingPass,
)

CTX = Context(allow_unregistered=True)
CTX.load_dialect(Arith)
CTX.load_dialect(Builtin)

CANONICALIZE_PASS = CanonicalizePass()
CONSTANT_FOLDING_SIMPLE_PASS = TestConstantFoldingPass()
CONSTANT_FOLDING_SPECIALISED_PASS = TestSpecialisedConstantFoldingPass()


def parse_module(context: Context, contents: str) -> ModuleOp:
    """Parse a MLIR file as a module."""
    parser = XdslParser(context, contents)
    return parser.parse_module()


class ConstantFolding:
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
        self.workload_constant_20 = ConstantFolding.WORKLOAD_CONSTANT_20.clone()

    def time_constant_folding_20(self) -> None:
        """Time canonicalizing constant folding for 20 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_20)

    def time_constant_folding_simple_20(self) -> None:
        """Time simple constant folding for 20 items."""
        CONSTANT_FOLDING_SIMPLE_PASS.apply(CTX, self.workload_constant_20)

    def time_constant_folding_specialised_20(self) -> None:
        """Time simple constant folding for 20 items."""
        CONSTANT_FOLDING_SPECIALISED_PASS.apply(CTX, self.workload_constant_20)

    def setup_constant_folding_100(self) -> None:
        """Setup the constant folding 100 items benchmark."""
        self.workload_constant_100 = ConstantFolding.WORKLOAD_CONSTANT_100.clone()

    def time_constant_folding_100(self) -> None:
        """Time canonicalizing constant folding for 100 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_100)

    def setup_constant_folding_1000(self) -> None:
        """Setup the constant folding 1000 items benchmark."""
        self.workload_constant_1000 = ConstantFolding.WORKLOAD_CONSTANT_1000.clone()

    def time_constant_folding_1000(self) -> None:
        """Time canonicalizing constant folding for 1000 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_1000)


class RewritingMicrobenchmarks:
    """Microbenchmarks for rewriting of constant folding."""

    WORKLOAD_CONSTANT_20 = parse_module(CTX, WorkloadBuilder.constant_folding(20))
    region: Region
    const_0: ConstantOp
    const_1: ConstantOp
    add_op: AddiOp
    sub_op: SubiOp
    integer_type: IntegerType

    def setup(self) -> None:
        """Setup the benchmarks."""
        self.region = ConstantFolding.WORKLOAD_CONSTANT_20.clone().body
        self.first_block = self.region.block
        ops = self.region.walk()
        self.const_0 = cast(ConstantOp, next(ops))
        self.const_1 = cast(ConstantOp, next(ops))
        self.add_op = cast(AddiOp, next(ops))
        self.add_op_def = AddiOp.get_irdl_definition()
        self.add_op_construct = VarIRConstruct.OPERAND
        self.add_op_result = self.add_op.result
        self.add_op_result_use = list(self.add_op_result.uses)[0]
        self.sub_op = SubiOp(self.const_1, self.const_0)
        self.sub_op_result = self.sub_op.result
        self.insert_point = InsertPoint.before(self.add_op)
        self.worklist = Worklist()
        self.worklist.push(self.sub_op)
        self.integer_type = IntegerType(64)
        self.integer_attr = IntegerAttr(0, 64)
        self.live_set = LiveSet()
        self.pattern_rewriter = PatternRewriter(self.add_op)


class PatternRewriting(RewritingMicrobenchmarks):
    """Microbenchmarks for general pattern rewriting of constant folding."""

    def time_region_walk(self) -> None:
        """Time `Region.walk`.

        Exercise walking over an IR region, including iterator overhead. This
        is used to populate the worklist for pattern rewriting.
        """
        for block in self.region.walk():
            assert block

    def time_worklist_push(self) -> None:
        """Time `Worklist.push`.

        Exercise adding an operation to the worklist, including lookups in the
        worklist map, relying on operation's `__hash__` function. This is used
        to populate the worklist for pattern rewriting.
        """
        self.worklist.push(self.add_op)

    def time_worklist_pop(self) -> None:
        """Time `Worklist.pop`.

        Exercise removing an operation from the worklist, from the worklist map,
        relying on operation's `__hash__` function. This is used to process
        items sequentially from the worklist as a stack.
        """
        self.worklist.pop()

    def time_get_trait(self) -> None:
        """Time `Operation.get_trait`.

        Exercise getting an operation's trait, including `isinstance` checks and
        the traits iterator. This is used frequently in pattern rewriting, both
        for matching and for the rewriting logic.
        """
        self.add_op.get_trait(HasCanonicalizationPatternsTrait)

    def time_insert_point_before(self) -> None:
        """Time `InsertPoint.before`.

        Measure time to create an `InsertPoint` from the block's parent block.
        """
        InsertPoint.before(self.add_op)

    def ignore_time_pattern_rewriter_insert_op(self) -> None:
        """Time `PatternRewriter.insert_op`.

        Exercise inserting an operation and running any required callbacks. This
        is used to effect the results of rewriting.
        """
        self.pattern_rewriter.insert_op((self.sub_op,), self.insert_point)

    def time_get_variadic_sizes(self) -> None:
        """Time `get_variadic_sizes`.

        Exercise getting the variadic size of an operation, including a
        significant amount of logic to check lengths and types. This is invoked
        whenever an operation argument is defined with `irdl_op_arg_definition`.
        """
        get_variadic_sizes(self.add_op, self.add_op_def, self.add_op_construct)


class Canonicalization(RewritingMicrobenchmarks):
    """Microbenchmarks for canonicalization rewriting of constant folding."""

    def ignore_time_operation_drop_all_references(self) -> None:
        """Time `Operation.drop_all_references`.

        Exercise dropping references to an operation, including removing
        references to its IR uses. This is used when removing unused operations.
        """
        self.add_op.drop_all_references()

    def time_ssavalue_replace_by(self) -> None:
        """Time `SSAValue.replace_by`.

        Exercise replacing an `SSAValue` and all its uses with another
        SSAValue. This is used when removing unused operations.
        """
        self.add_op_result.replace_by(self.sub_op_result)

    def ignore_time_irwithuses_remove_use(self) -> None:
        """Time `IRWithUses.remove_use`.

        Exercise removing references to IR uses. This is used when removing
        unused operations.
        """
        self.add_op_result.remove_use(self.add_op_result_use)

    def time_irwithuses_add_use(self) -> None:
        """Time `IRWithUses.add_use`.

        Exercise adding references to IR uses.
        """
        self.sub_op_result.add_use(self.add_op_result_use)

    def time_ssavalue_name_hint(self) -> None:
        """Time `SSAValue.namehint`.

        Exercise changing an `SSAValue` name hint, which performs regex
        validation at runtime. This is used when removing unused operations.
        """
        self.add_op_result.name_hint = "valid_name"

    def time_handle_operation_removal(self) -> None:
        """Time `PatternRewriter.handle_operation_removal`.

        Exercise removing an operation recursively from regions, and from the
        pattern rewriter worklist. This is used when removing unused operations.
        """
        self.pattern_rewriter.handle_operation_removal(self.add_op)

    def ignore_time_block_detach_op(self) -> None:
        """Time `Block.detach_op`.

        Exercise detaching an operation from a block, including fixing the
        block's doubly linked list. This is used when removing unused
        operations.
        """
        self.region.block.detach_op(self.add_op)

    def time_ssavalue_erase(self) -> None:
        """Time `SSAValue.erase`.

        Exercise erasing an 'SSAValue'. This is used when removing unused
        operations.
        """
        self.add_op_result.erase(safe_erase=False)

    def time_integer_attr_creation(self) -> None:
        """Time `IntegerAttr.__init__`.

        Exercise instantiating an integer attribute, including getting a
        normalised value and constructing the class. This is used when creating
        a new constant operation for folding.
        """
        IntegerAttr(0, 64)

    def time_integer_type_normalized_value(self) -> None:
        """Time `IntegerType.normalized_value`.

        Exercise getting the normalised value for an integer. This is used when creating
        a new constant operation for folding.
        """
        self.integer_type.normalized_value(0)

    def time_integer_attr_verify(self) -> None:
        """Time `IntegerAttr._verify`.

        Exercise verifying an integer attribute. This is used when creating a
        new constant operation for folding.
        """
        self.integer_attr._verify()  # pyright: ignore[reportPrivateUsage]

    def time_operation_create(self) -> None:
        """Time `AddiOp.__init__`.

        Exercising instantiating an add operation of two constant values.
        """
        AddiOp(self.const_1, self.const_0)

    def time_const_evaluate_operand(self) -> None:
        """Time `const_evaluate_operand`.

        Exercise getting the value from a constant.
        """
        const_evaluate_operand(self.add_op.lhs)


class RemoveUnused(RewritingMicrobenchmarks):
    """Microbenchmarks for unused code removal rewriting of constant folding."""

    def time_is_trivially_dead(self) -> None:
        """Time `is_trivially_dead`.

        Exercise checking if an operation has no observable effect. This is
        mostly checking if it would be trivially dead, and alse checking its
        results have no uses.
        """
        is_trivially_dead(self.add_op)

    def time_would_be_trivially_dead(self) -> None:
        """Time `would_be_trivially_dead`.

        Exercise checking if an operation would be dead if all its results were
        dead. This is around half trait checks and half getting result only
        effects.
        """
        would_be_trivially_dead(self.add_op)

    def time_result_only_effects(self) -> None:
        """Time `result_only_effects`.

        Exercise checking if an operation has no observable effect beyond its
        return value. This is mostly getting its effects.
        """
        result_only_effects(self.add_op)

    def time_operation_get_traits_of_type(self) -> None:
        """Time `Operation.get_traits_of_type`.

        Exercise getting all the traits from an operatino of a give type. This
        is mostly `isinstance` checks.
        """
        self.add_op.get_traits_of_type(MemoryEffect)


class RegionDCE(RewritingMicrobenchmarks):
    """Microbenchmarks for region dead-code elimination of constant folding."""

    def time_post_order_iterator(self) -> None:
        """Time `PostOrderIterator`.

        Exercise constructing and using a post-order iterator over a block. This
        is used for dead code elimination.
        """
        post_order_iterator = PostOrderIterator(self.first_block)
        for block in post_order_iterator:
            assert block

    def time_liveset_set_live(self) -> None:
        """Time `LiveSet.set_live`.

        Exercise adding an item to the live set. This is used for dead code
        elimination.
        """
        self.live_set.set_live(self.add_op)

    def time_liveset_delete_dead(self) -> None:
        """Time `LiveSet.delete_dead`.

        Exercise deleting operations not in the live set for a region. This is
        used for dead code elimination.
        """
        self.live_set.delete_dead(self.region, None)


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    GENERAL = PatternRewriting()
    CONSTANT_FOLDING = ConstantFolding()
    CANONICALIZATION = Canonicalization()
    REMOVE_UNUSED = RemoveUnused()
    REGION_DCE = RegionDCE()
    profile(
        {
            "ConstantFolding.20": Benchmark(
                CONSTANT_FOLDING.time_constant_folding_20,
                CONSTANT_FOLDING.setup_constant_folding_20,
            ),
            "ConstantFoldingSimple.20": Benchmark(
                CONSTANT_FOLDING.time_constant_folding_simple_20,
                CONSTANT_FOLDING.setup_constant_folding_20,
            ),
            "ConstantFoldingSpecialised.20": Benchmark(
                CONSTANT_FOLDING.time_constant_folding_specialised_20,
                CONSTANT_FOLDING.setup_constant_folding_20,
            ),
            "ConstantFolding.100": Benchmark(
                CONSTANT_FOLDING.time_constant_folding_100,
                CONSTANT_FOLDING.setup_constant_folding_100,
            ),
            "ConstantFolding.1000": Benchmark(
                CONSTANT_FOLDING.time_constant_folding_1000,
                CONSTANT_FOLDING.setup_constant_folding_1000,
            ),
            # ================================================================ #
            "General.region_walk": Benchmark(GENERAL.time_region_walk, GENERAL.setup),
            "General.worklist_push": Benchmark(
                GENERAL.time_worklist_push,
                GENERAL.setup,
            ),
            "General.worklist_pop": Benchmark(
                GENERAL.time_worklist_pop,
                GENERAL.setup,
            ),
            "General.get_trait": Benchmark(
                GENERAL.time_get_trait,
                GENERAL.setup,
            ),
            "General.insert_point_before": Benchmark(
                GENERAL.time_insert_point_before,
                GENERAL.setup,
            ),
            "General.pattern_rewriter_insert_op": Benchmark(
                GENERAL.ignore_time_pattern_rewriter_insert_op,
                GENERAL.setup,
            ),
            "General.get_variadic_sizes": Benchmark(
                GENERAL.time_get_variadic_sizes, GENERAL.setup
            ),
            "Canonicalization.operation_drop_all_references": Benchmark(
                CANONICALIZATION.ignore_time_operation_drop_all_references,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.ssavalue_replace_by": Benchmark(
                CANONICALIZATION.time_ssavalue_replace_by,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.irwithuses_remove_use": Benchmark(
                CANONICALIZATION.ignore_time_irwithuses_remove_use,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.irwithuses_add_use": Benchmark(
                CANONICALIZATION.time_irwithuses_add_use,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.ssavalue_name_hint": Benchmark(
                CANONICALIZATION.time_ssavalue_name_hint,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.handle_operation_removal": Benchmark(
                CANONICALIZATION.time_handle_operation_removal,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.block_detach_op": Benchmark(
                CANONICALIZATION.ignore_time_block_detach_op,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.ssavalue_erase": Benchmark(
                CANONICALIZATION.time_ssavalue_erase,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.integer_attr_creation": Benchmark(
                CANONICALIZATION.time_integer_attr_creation,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.integer_type_normalized_value": Benchmark(
                CANONICALIZATION.time_integer_type_normalized_value,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.integer_attr_verify": Benchmark(
                CANONICALIZATION.time_integer_attr_verify,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.operation_create": Benchmark(
                CANONICALIZATION.time_operation_create,
                CANONICALIZATION.setup,
            ),
            "Canonicalization.const_evaluate_operand": Benchmark(
                CANONICALIZATION.time_const_evaluate_operand,
                CANONICALIZATION.setup,
            ),
            "RemoveUnused.is_trivially_dead": Benchmark(
                REMOVE_UNUSED.time_is_trivially_dead,
                REMOVE_UNUSED.setup,
            ),
            "RemoveUnused.would_be_trivially_dead": Benchmark(
                REMOVE_UNUSED.time_would_be_trivially_dead,
                REMOVE_UNUSED.setup,
            ),
            "RemoveUnused.result_only_effects": Benchmark(
                REMOVE_UNUSED.time_result_only_effects,
                REMOVE_UNUSED.setup,
            ),
            "RemoveUnused.operation_get_traits_of_type": Benchmark(
                REMOVE_UNUSED.time_operation_get_traits_of_type,
                REMOVE_UNUSED.setup,
            ),
            "RegionDCE.post_order_iterator": Benchmark(
                REGION_DCE.time_post_order_iterator,
                REGION_DCE.setup,
            ),
            "RegionDCE.liveset_set_live": Benchmark(
                REGION_DCE.time_liveset_set_live,
                REGION_DCE.setup,
            ),
            "RegionDCE.liveset_delete_dead": Benchmark(
                REGION_DCE.time_liveset_delete_dead,
                REGION_DCE.setup,
            ),
        }
    )
