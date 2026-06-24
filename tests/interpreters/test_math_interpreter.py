from math import exp, log, sqrt

import pytest

from xdsl.dialects import test
from xdsl.dialects.builtin import ModuleOp, f64
from xdsl.dialects.math import ExpOp, LogOp, SqrtOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.math import MathFunctions


@pytest.fixture
def operand_op() -> test.TestOp:
    return test.TestOp(result_types=[f64])


@pytest.fixture
def interpreter() -> Interpreter:
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MathFunctions())
    return interpreter


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.5, 0.25])
def test_exp(interpreter: Interpreter, operand_op: test.TestOp, value: float):
    op = ExpOp(operand_op)

    ret = interpreter.run_op(op, (value,))

    assert len(ret) == 1
    assert ret[0] == exp(value)


@pytest.mark.parametrize("value", [0.0, 1.0, 4.0, 2.25])
def test_sqrt(interpreter: Interpreter, operand_op: test.TestOp, value: float):
    op = SqrtOp(operand_op)

    ret = interpreter.run_op(op, (value,))

    assert len(ret) == 1
    assert ret[0] == sqrt(value)


@pytest.mark.parametrize("value", [1.0, 2.718281828, 10.0, 0.5])
def test_log(interpreter: Interpreter, operand_op: test.TestOp, value: float):
    op = LogOp(operand_op)

    ret = interpreter.run_op(op, (value,))

    assert len(ret) == 1
    assert ret[0] == log(value)
