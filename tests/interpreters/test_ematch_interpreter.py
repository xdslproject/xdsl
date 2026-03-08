from dataclasses import dataclass
from typing import Any, cast

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, ematch, equivalence, pdl, test
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ematch import EmatchFunctions
from xdsl.interpreters.eqsat_pdl_interp import NonPropagatingDataFlowSolver
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Operation, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


def test_get_class_vals():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(EmatchFunctions())

    # Create a test operation with results
    v0 = create_ssa_value(i32)
    v1 = create_ssa_value(i32)
    c = equivalence.ClassOp(v0, v1).result

    v2 = create_ssa_value(i32)

    assert interpreter.run_op(
        ematch.GetClassValsOp(create_ssa_value(pdl.ValueType())), (v1,)
    ) == ((v1,),)
    assert interpreter.run_op(
        ematch.GetClassValsOp(create_ssa_value(pdl.ValueType())), (c,)
    ) == ((v0, v1),)
    assert interpreter.run_op(
        ematch.GetClassValsOp(create_ssa_value(pdl.ValueType())), (v2,)
    ) == ((v2,),)
    assert interpreter.run_op(
        ematch.GetClassValsOp(create_ssa_value(pdl.ValueType())), (None,)
    ) == ((None,),)
    block_arg = Block(arg_types=(i32,)).args[0]
    assert interpreter.run_op(
        ematch.GetClassValsOp(create_ssa_value(pdl.ValueType())), (block_arg,)
    ) == ((block_arg,),)


def test_get_class_representative():
    ematch_funcs = EmatchFunctions()
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ematch_funcs)

    v0 = create_ssa_value(i32)
    v1 = create_ssa_value(i32)
    classop = equivalence.ClassOp(v0, v1)

    # Class result → first operand (representative)
    assert interpreter.run_op(
        ematch.GetClassRepresentativeOp(create_ssa_value(pdl.ValueType())),
        (classop.result,),
    ) == (v0,)

    # Plain value → itself
    v2 = create_ssa_value(i32)
    assert interpreter.run_op(
        ematch.GetClassRepresentativeOp(create_ssa_value(pdl.ValueType())),
        (v2,),
    ) == (v2,)

    assert interpreter.run_op(
        ematch.GetClassRepresentativeOp(create_ssa_value(pdl.ValueType())),
        (None,),
    ) == (None,)

    block_arg = Block(arg_types=(i32,)).args[0]
    assert interpreter.run_op(
        ematch.GetClassRepresentativeOp(create_ssa_value(pdl.ValueType())), (block_arg,)
    ) == (block_arg,)


def test_get_class_result():
    ematch_funcs = EmatchFunctions()
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ematch_funcs)

    v0 = create_ssa_value(i32)
    classop = equivalence.ClassOp(v0)

    # v0 has one use (the ClassOp) → returns ClassOp's result
    assert interpreter.run_op(
        ematch.GetClassResultOp(create_ssa_value(pdl.ValueType())),
        (v0,),
    ) == (classop.result,)

    # v1 has no ClassOp user → returns itself
    v1 = create_ssa_value(i32)
    assert interpreter.run_op(
        ematch.GetClassResultOp(create_ssa_value(pdl.ValueType())),
        (v1,),
    ) == (v1,)

    # None → returns None
    assert interpreter.run_op(
        ematch.GetClassResultOp(create_ssa_value(pdl.ValueType())),
        (None,),
    ) == (None,)

    # v2 has one use that is *not* a ClassOp → returns itself
    v2 = create_ssa_value(i32)
    _ = test.TestOp(operands=(v2,))  # gives v2 exactly one (non-ClassOp) use
    assert interpreter.run_op(
        ematch.GetClassResultOp(create_ssa_value(pdl.ValueType())),
        (v2,),
    ) == (v2,)


def test_get_class_results():
    ematch_funcs = EmatchFunctions()
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ematch_funcs)

    v0 = create_ssa_value(i32)
    v1 = create_ssa_value(i32)
    classop0 = equivalence.ClassOp(v0)
    classop1 = equivalence.ClassOp(v1)

    assert interpreter.run_op(
        ematch.GetClassResultsOp(create_ssa_value(pdl.RangeType(pdl.ValueType()))),
        ((v0, v1),),
    ) == ((classop0.result, classop1.result),)

    # Value without ClassOp user → returns itself
    v2 = create_ssa_value(i32)
    assert interpreter.run_op(
        ematch.GetClassResultsOp(create_ssa_value(pdl.RangeType(pdl.ValueType()))),
        ((v2,),),
    ) == ((v2,),)

    # Single None → returns (None,)
    assert interpreter.run_op(
        ematch.GetClassResultsOp(create_ssa_value(pdl.RangeType(pdl.ValueType()))),
        (None,),
    ) == ((),)

    # Tuple of Nones → returns tuple of Nones
    assert interpreter.run_op(
        ematch.GetClassResultsOp(create_ssa_value(pdl.RangeType(pdl.ValueType()))),
        ((None, None),),
    ) == ((None, None),)

    # Mixed: ClassOp value and None
    v3 = create_ssa_value(i32)
    classop2 = equivalence.ClassOp(v3)
    assert interpreter.run_op(
        ematch.GetClassResultsOp(create_ssa_value(pdl.RangeType(pdl.ValueType()))),
        ((v3, None),),
    ) == ((classop2.result, None),)

    # Value with one non-ClassOp use → returns itself
    v4 = create_ssa_value(i32)
    _ = test.TestOp(operands=(v4,))  # gives v4 exactly one non-ClassOp use
    assert interpreter.run_op(
        ematch.GetClassResultsOp(create_ssa_value(pdl.RangeType(pdl.ValueType()))),
        ((v4,),),
    ) == ((v4,),)


def _make_interpreter_with_rewriter():
    """Helper to set up an interpreter with a rewriter for tests that modify IR."""
    ematch_funcs = EmatchFunctions()
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ematch_funcs)

    testmodule = ModuleOp(Region([Block()]))
    block = testmodule.body.first_block
    assert block is not None
    with ImplicitBuilder(block):
        root = test.TestOp()

    rewriter = PatternRewriter(root)
    PDLInterpFunctions.set_rewriter(interpreter, rewriter)

    return interpreter, ematch_funcs, block


def test_get_or_create_class_val_is_class_result():
    """If val is defined by a ClassOp, return that ClassOp."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        classop = equivalence.ClassOp(create_ssa_value(i32))

    result = ematch_funcs.get_or_create_class(interpreter, classop.result)
    assert result is classop


def test_get_or_create_class_val_has_classop_user():
    """If val has exactly one use and that use is a ClassOp, return it."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).res[0]
        classop = equivalence.ClassOp(v0)

    result = ematch_funcs.get_or_create_class(interpreter, v0)
    assert result is classop


def test_get_or_create_class_creates_new_class_for_opresult():
    """If val is an OpResult without a ClassOp user, create a new ClassOp."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).res[0]
        _user = test.TestOp((v0,))

    result = ematch_funcs.get_or_create_class(interpreter, v0)
    assert isinstance(result, equivalence.ClassOp)
    assert v0 in result.operands
    assert ematch_funcs.eclass_union_find.find(result) is result


def test_get_or_create_class_creates_new_class_for_block_arg():
    """If val is a block argument without a ClassOp user, create a new ClassOp."""
    ematch_funcs = EmatchFunctions()
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(ematch_funcs)

    testmodule = ModuleOp(Region([Block(arg_types=(i32,))]))
    block = testmodule.body.first_block
    assert block is not None
    block_arg = block.args[0]
    with ImplicitBuilder(block):
        root = test.TestOp()

    rewriter = PatternRewriter(root)
    PDLInterpFunctions.set_rewriter(interpreter, rewriter)

    result = ematch_funcs.get_or_create_class(interpreter, block_arg)
    assert isinstance(result, equivalence.ClassOp)
    assert block_arg in result.operands
    assert ematch_funcs.eclass_union_find.find(result) is result


def test_union_val():
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]

    interpreter.run_op(
        ematch.UnionOp(
            create_ssa_value(pdl.ValueType()), create_ssa_value(pdl.ValueType())
        ),
        (v0, v1),
    )

    # After union, both values should be operands of the same ClassOp
    eclass_a = ematch_funcs.get_or_create_class(interpreter, v0)
    eclass_b = ematch_funcs.get_or_create_class(interpreter, v1)
    assert eclass_a is eclass_b
    assert set(eclass_a.operands) == {v0, v1}


def test_eclass_union_same_class():
    """Union of the same eclass with itself is a no-op."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        eclass = equivalence.ClassOp(v0, res_type=i32)

    ematch_funcs.eclass_union_find.add(eclass)

    result = ematch_funcs.eclass_union(interpreter, eclass, eclass)
    assert result is False
    assert len(ematch_funcs.worklist) == 0


def test_eclass_union_two_regular_classes():
    """Union of two regular eclasses merges operands and replaces one."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]
        eclass_a = equivalence.ClassOp(v0, res_type=i32)
        eclass_b = equivalence.ClassOp(v1, res_type=i32)

    ematch_funcs.eclass_union_find.add(eclass_a)
    ematch_funcs.eclass_union_find.add(eclass_b)

    result = ematch_funcs.eclass_union(interpreter, eclass_a, eclass_b)
    assert result is True

    canonical = ematch_funcs.eclass_union_find.find(eclass_a)
    assert ematch_funcs.eclass_union_find.find(eclass_b) is canonical
    assert set(canonical.operands) == {v0, v1}


def test_eclass_union_constant_with_regular():
    """Union of ConstantClassOp with regular ClassOp keeps the constant as canonical."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        const_op = arith.ConstantOp(IntegerAttr(1, i32))
        regular_op = test.TestOp(result_types=(i32,))

        const_eclass = equivalence.ConstantClassOp(const_op.result)
        regular_eclass = equivalence.ClassOp(regular_op.results[0], res_type=i32)

    ematch_funcs.eclass_union_find.add(const_eclass)
    ematch_funcs.eclass_union_find.add(regular_eclass)

    result = ematch_funcs.eclass_union(interpreter, const_eclass, regular_eclass)
    assert result is True

    canonical = ematch_funcs.eclass_union_find.find(const_eclass)
    assert isinstance(canonical, equivalence.ConstantClassOp)
    assert canonical.value == IntegerAttr(1, i32)
    assert set(canonical.operands) == {const_op.result, regular_op.results[0]}


def test_eclass_union_regular_with_constant():
    """Union with constant as second argument still keeps constant as canonical."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        regular_op = test.TestOp(result_types=(i32,))
        const_op = arith.ConstantOp(IntegerAttr(42, i32))

        regular_eclass = equivalence.ClassOp(regular_op.results[0], res_type=i32)
        const_eclass = equivalence.ConstantClassOp(const_op.result)

    ematch_funcs.eclass_union_find.add(regular_eclass)
    ematch_funcs.eclass_union_find.add(const_eclass)

    result = ematch_funcs.eclass_union(interpreter, regular_eclass, const_eclass)
    assert result is True

    canonical = ematch_funcs.eclass_union_find.find(regular_eclass)
    assert isinstance(canonical, equivalence.ConstantClassOp)
    assert canonical is const_eclass


def test_eclass_union_two_same_constants():
    """Union of two ConstantClassOps with the same value succeeds."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        const1 = arith.ConstantOp(IntegerAttr(5, i32))
        const2 = arith.ConstantOp(IntegerAttr(5, i32))

        const_eclass1 = equivalence.ConstantClassOp(const1.result)
        const_eclass2 = equivalence.ConstantClassOp(const2.result)

    ematch_funcs.eclass_union_find.add(const_eclass1)
    ematch_funcs.eclass_union_find.add(const_eclass2)

    result = ematch_funcs.eclass_union(interpreter, const_eclass1, const_eclass2)
    assert result is True

    canonical = ematch_funcs.eclass_union_find.find(const_eclass1)
    assert isinstance(canonical, equivalence.ConstantClassOp)
    assert set(canonical.operands) == {const1.result, const2.result}


def test_eclass_union_two_different_constants_fails():
    """Union of two ConstantClassOps with different values raises AssertionError."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        const1 = arith.ConstantOp(IntegerAttr(1, i32))
        const2 = arith.ConstantOp(IntegerAttr(2, i32))

        const_eclass1 = equivalence.ConstantClassOp(const1.result)
        const_eclass2 = equivalence.ConstantClassOp(const2.result)

    ematch_funcs.eclass_union_find.add(const_eclass1)
    ematch_funcs.eclass_union_find.add(const_eclass2)

    with pytest.raises(
        AssertionError, match="Trying to union two different constant eclasses."
    ):
        ematch_funcs.eclass_union(interpreter, const_eclass1, const_eclass2)


def test_eclass_union_removes_uses_from_known_ops():
    """Uses of the replaced eclass are removed from known_ops before replacement."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]
        eclass_a = equivalence.ClassOp(v0, res_type=i32)
        eclass_b = equivalence.ClassOp(v1, res_type=i32)

        # Create an operation that uses eclass_b's result — it should be
        # removed from known_ops when eclass_b is replaced.
        user_op = test.TestOp((eclass_b.result,), result_types=(i32,))

    ematch_funcs.eclass_union_find.add(eclass_a)
    ematch_funcs.eclass_union_find.add(eclass_b)
    ematch_funcs.known_ops[user_op] = user_op

    ematch_funcs.eclass_union(interpreter, eclass_a, eclass_b)

    # user_op used eclass_b.result, so it must have been popped from known_ops
    assert user_op not in ematch_funcs.known_ops


def test_eclass_union_deduplicates_operands():
    """When the same value is an operand of both eclasses, it appears only once after union."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        shared = test.TestOp(result_types=(i32,)).results[0]
        extra = test.TestOp(result_types=(i32,)).results[0]
        eclass_a = equivalence.ClassOp(shared, res_type=i32)
        eclass_b = equivalence.ClassOp(shared, extra, res_type=i32)

    ematch_funcs.eclass_union_find.add(eclass_a)
    ematch_funcs.eclass_union_find.add(eclass_b)

    ematch_funcs.eclass_union(interpreter, eclass_a, eclass_b)

    canonical = ematch_funcs.eclass_union_find.find(eclass_a)
    # shared should appear only once
    operand_list = list(canonical.operands)
    assert operand_list.count(shared) == 1
    assert set(canonical.operands) == {shared, extra}


def test_eclass_union_meets_analysis_states():
    """Analysis lattice states are met when eclasses are unioned."""
    from typing_extensions import Self

    from xdsl.analysis.dataflow import DataFlowSolver
    from xdsl.analysis.sparse_analysis import (
        AbstractLatticeValue,
        Lattice,
        SparseForwardDataFlowAnalysis,
    )

    @dataclass(frozen=True)
    class TestLatticeValue(AbstractLatticeValue):
        value: int

        @classmethod
        def initial_value(cls) -> Self:
            return cls(0)

        def meet(self, other: "TestLatticeValue") -> "TestLatticeValue":
            return TestLatticeValue(min(self.value, other.value))

        def join(self, other: "TestLatticeValue") -> "TestLatticeValue":
            return TestLatticeValue(max(self.value, other.value))

    class TestLattice(Lattice[TestLatticeValue]):
        value_cls = TestLatticeValue

    class TestAnalysis(SparseForwardDataFlowAnalysis[TestLattice]):
        def __init__(self, solver: DataFlowSolver):
            super().__init__(solver, TestLattice)

        def visit_operation_impl(
            self,
            op: Operation,
            operands: list[TestLattice],
            results: list[TestLattice],
        ) -> None:
            pass

        def set_to_entry_state(self, lattice: TestLattice) -> None:
            pass

    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    ctx = PDLInterpFunctions.get_ctx(interpreter)
    solver = NonPropagatingDataFlowSolver(ctx)
    analysis = TestAnalysis(solver)
    ematch_funcs.analyses.append(
        cast(SparseForwardDataFlowAnalysis[Lattice[Any]], analysis)
    )

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]
        eclass_a = equivalence.ClassOp(v0, res_type=i32)
        eclass_b = equivalence.ClassOp(v1, res_type=i32)

    ematch_funcs.eclass_union_find.add(eclass_a)
    ematch_funcs.eclass_union_find.add(eclass_b)

    # Set lattice values: a=10, b=3 → meet = min(10, 3) = 3
    lattice_a = analysis.get_lattice_element(eclass_a.result)
    lattice_a._value = TestLatticeValue(10)  # pyright: ignore[reportPrivateUsage]
    lattice_b = analysis.get_lattice_element(eclass_b.result)
    lattice_b._value = TestLatticeValue(3)  # pyright: ignore[reportPrivateUsage]

    ematch_funcs.eclass_union(interpreter, eclass_a, eclass_b)

    # The surviving eclass (a) should have the met value
    canonical = ematch_funcs.eclass_union_find.find(eclass_a)
    result_lattice = analysis.get_lattice_element(canonical.result)
    assert result_lattice.value.value == 3


def test_eclass_union_removes_uses_not_in_known_ops():
    """Operations using the replaced eclass that are NOT in known_ops should not cause errors."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]
        eclass_a = equivalence.ClassOp(v0, res_type=i32)
        eclass_b = equivalence.ClassOp(v1, res_type=i32)

        # Create an operation that uses eclass_b's result but do NOT add it to known_ops
        user_op = test.TestOp((eclass_b.result,), result_types=(i32,))

    ematch_funcs.eclass_union_find.add(eclass_a)
    ematch_funcs.eclass_union_find.add(eclass_b)

    # user_op is intentionally not in known_ops
    assert user_op not in ematch_funcs.known_ops

    # Should succeed without error
    ematch_funcs.eclass_union(interpreter, eclass_a, eclass_b)

    # user_op should still not be in known_ops
    assert user_op not in ematch_funcs.known_ops


def test_union_val_same_value():
    """Union of a value with itself is a no-op."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]

    ematch_funcs.union_val(interpreter, v0, v0)

    # No worklist entries should be created
    assert len(ematch_funcs.worklist) == 0


def test_union_val_already_same_eclass():
    """Union of two values already in the same eclass is a no-op (via eclass_union)."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]
        eclass = equivalence.ClassOp(v0, v1, res_type=i32)

    ematch_funcs.eclass_union_find.add(eclass)

    ematch_funcs.union_val(interpreter, v0, v1)

    # Both already in the same eclass, so no worklist entries
    assert len(ematch_funcs.worklist) == 0


def test_run_union_operation_and_value_range():
    """Union of an operation with a value range merges results pairwise."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        op = test.TestOp(result_types=(i32, i32))
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]

    interpreter.run_op(
        ematch.UnionOp(
            create_ssa_value(pdl.OperationType()),
            create_ssa_value(pdl.RangeType(pdl.ValueType())),
        ),
        (op, (v0, v1)),
    )

    eclass_r0 = ematch_funcs.get_or_create_class(interpreter, op.results[0])
    eclass_v0 = ematch_funcs.get_or_create_class(interpreter, v0)
    assert ematch_funcs.eclass_union_find.find(
        eclass_r0
    ) is ematch_funcs.eclass_union_find.find(eclass_v0)

    eclass_r1 = ematch_funcs.get_or_create_class(interpreter, op.results[1])
    eclass_v1 = ematch_funcs.get_or_create_class(interpreter, v1)
    assert ematch_funcs.eclass_union_find.find(
        eclass_r1
    ) is ematch_funcs.eclass_union_find.find(eclass_v1)


def test_run_union_two_value_ranges():
    """Union of two value ranges merges values pairwise."""
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        v0 = test.TestOp(result_types=(i32,)).results[0]
        v1 = test.TestOp(result_types=(i32,)).results[0]
        v2 = test.TestOp(result_types=(i32,)).results[0]
        v3 = test.TestOp(result_types=(i32,)).results[0]

    interpreter.run_op(
        ematch.UnionOp(
            create_ssa_value(pdl.RangeType(pdl.ValueType())),
            create_ssa_value(pdl.RangeType(pdl.ValueType())),
        ),
        ((v0, v1), (v2, v3)),
    )

    eclass_v0 = ematch_funcs.get_or_create_class(interpreter, v0)
    eclass_v2 = ematch_funcs.get_or_create_class(interpreter, v2)
    assert ematch_funcs.eclass_union_find.find(
        eclass_v0
    ) is ematch_funcs.eclass_union_find.find(eclass_v2)

    eclass_v1 = ematch_funcs.get_or_create_class(interpreter, v1)
    eclass_v3 = ematch_funcs.get_or_create_class(interpreter, v3)
    assert ematch_funcs.eclass_union_find.find(
        eclass_v1
    ) is ematch_funcs.eclass_union_find.find(eclass_v3)


def test_run_union_unsupported_types():
    """Union with unsupported argument types raises InterpretationError."""
    interpreter, _ematch_funcs, _block = _make_interpreter_with_rewriter()

    with pytest.raises(InterpretationError, match="unsupported argument types"):
        interpreter.run_op(
            ematch.UnionOp(
                create_ssa_value(pdl.ValueType()),
                create_ssa_value(pdl.ValueType()),
            ),
            ("not_a_value", 42),
        )
