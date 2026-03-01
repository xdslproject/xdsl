from xdsl.builder import ImplicitBuilder
from xdsl.dialects import ematch, equivalence, pdl, test
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ematch import EmatchFunctions
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Region
from xdsl.pattern_rewriter import PatternRewriter
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


def test_dedup():
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        # Create two structurally identical operations
        existing_op = test.TestOp(result_types=(i32,))
        new_op = test.TestOp(result_types=(i32,))

    ematch_funcs.known_ops[existing_op] = existing_op

    # Dedup new_op → should return existing_op and erase new_op
    result = interpreter.run_op(
        ematch.DedupOp(create_ssa_value(pdl.OperationType())),
        (new_op,),
    )
    assert result == (existing_op,)

    # new_op should no longer be in the block
    assert new_op.parent is None


def test_dedup_no_duplicate():
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        op = test.TestOp(result_types=(i32,))

    # No existing equivalent in known_ops
    result = interpreter.run_op(
        ematch.DedupOp(create_ssa_value(pdl.OperationType())),
        (op,),
    )
    assert result == (op,)
    assert op in ematch_funcs.known_ops


def test_rebuild():
    interpreter, ematch_funcs, block = _make_interpreter_with_rewriter()

    with ImplicitBuilder(block):
        # Create an eclass with one value
        v0 = test.TestOp(result_types=(i32,)).results[0]

    eclass_c = ematch_funcs.get_or_create_class(interpreter, v0)

    with ImplicitBuilder(block):
        # Create two identical operations using the eclass result
        op1 = test.TestOp(operands=(eclass_c.result,), result_types=(i32,))
        op2 = test.TestOp(operands=(eclass_c.result,), result_types=(i32,))

    # Create eclasses for op1 and op2's results
    eclass_a = ematch_funcs.get_or_create_class(interpreter, op1.results[0])
    eclass_b = ematch_funcs.get_or_create_class(interpreter, op2.results[0])
    assert eclass_a is not eclass_b

    # Put eclass_c on the worklist and rebuild
    ematch_funcs.worklist.append(eclass_c)
    ematch_funcs.rebuild(interpreter)

    # After rebuild, eclass_a and eclass_b should be merged
    # (because op1 and op2 are structurally identical)
    assert ematch_funcs.eclass_union_find.find(
        eclass_a
    ) is ematch_funcs.eclass_union_find.find(eclass_b)
