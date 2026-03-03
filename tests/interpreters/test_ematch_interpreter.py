from xdsl.dialects import ematch, equivalence, pdl, test
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ematch import EmatchFunctions
from xdsl.ir import Block
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
