from xdsl.dialects import riscv, riscv_cf
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, Successor
from xdsl.interpreters.riscv_cf import RiscvCfFunctions
from xdsl.ir import Block
from xdsl.utils.test_value import create_ssa_value

register = riscv.Registers.UNALLOCATED_INT
module_op = ModuleOp([])


def test_j_op():
    interpreter = Interpreter(module_op)
    riscv_cf_functions = RiscvCfFunctions()
    interpreter.register_implementations(riscv_cf_functions)

    a = create_ssa_value(register)
    b = create_ssa_value(register)

    successor = Block(arg_types=(register, register))

    j_op = riscv_cf.JOp((a, b), successor)

    res = riscv_cf_functions.run_j(interpreter, j_op, (1, 2))

    assert res.values == ()
    assert res.terminator_value is not None
    assert isinstance(res.terminator_value, Successor)
    assert res.terminator_value.block is successor
    assert res.terminator_value.args == (1, 2)


def test_branch_op():
    interpreter = Interpreter(module_op)
    riscv_cf_functions = RiscvCfFunctions()
    interpreter.register_implementations(riscv_cf_functions)

    a = create_ssa_value(register)
    b = create_ssa_value(register)

    successor = Block(arg_types=(register, register))

    branch_op = riscv_cf.BranchOp((a, b), successor)

    res = riscv_cf_functions.run_branch(interpreter, branch_op, (1, 2))

    assert res.values == ()
    assert res.terminator_value is not None
    assert isinstance(res.terminator_value, Successor)
    assert res.terminator_value.block is successor
    assert res.terminator_value.args == (1, 2)


def test_beq_op():
    interpreter = Interpreter(module_op)
    riscv_cf_functions = RiscvCfFunctions()
    interpreter.register_implementations(riscv_cf_functions)

    lhs = create_ssa_value(register)
    rhs = create_ssa_value(register)
    t0 = create_ssa_value(register)
    t1 = create_ssa_value(register)
    e0 = create_ssa_value(register)

    then_block = Block(arg_types=(register, register))
    else_block = Block(arg_types=(register,))

    beq_op = riscv_cf.BeqOp(lhs, rhs, (t0, t1), (e0,), then_block, else_block)

    interpreter.push_scope("equal")
    interpreter.set_values(
        (
            (lhs, 1),
            (rhs, 1),
            (t0, 3),
            (t1, 4),
            (e0, 5),
        )
    )

    res_equal = riscv_cf_functions.run_beq(interpreter, beq_op, (1, 1, 3, 4, 5))

    assert res_equal.values == ()
    assert res_equal.terminator_value is not None
    assert isinstance(res_equal.terminator_value, Successor)
    assert res_equal.terminator_value.block is then_block
    assert res_equal.terminator_value.args == (3, 4)

    interpreter.pop_scope()

    interpreter.push_scope("not_equal")
    interpreter.set_values(
        (
            (lhs, 1),
            (rhs, 2),
            (t0, 3),
            (t1, 4),
            (e0, 5),
        )
    )

    res_equal = riscv_cf_functions.run_beq(interpreter, beq_op, (1, 2, 3, 4, 5))

    assert res_equal.values == ()
    assert res_equal.terminator_value is not None
    assert isinstance(res_equal.terminator_value, Successor)
    assert res_equal.terminator_value.block is else_block
    assert res_equal.terminator_value.args == (5,)
