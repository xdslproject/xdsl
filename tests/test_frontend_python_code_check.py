import ast

import pytest

from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.python_code_check import CheckAndInlineConstants


def test_const_correctly_evaluated_I():
    src = """
a: Const[i32] = 2 ** 5
x = a
"""
    stmts = ast.parse(src).body
    CheckAndInlineConstants.run(stmts, __file__)
    assert ast.unparse(stmts).endswith("x = 32")


def test_const_correctly_evaluated_II():
    src = """
a: Const[i32] = 4
x: i64 = a + 2
"""
    stmts = ast.parse(src).body
    CheckAndInlineConstants.run(stmts, __file__)
    assert ast.unparse(stmts).endswith("x: i64 = 4 + 2")


def test_const_correctly_evaluated_III():
    src = """
a: Const[i32] = 4
b: Const[i32] = len([1, 2, 3, 4])
x: Const[i32] = a + b
y = x
"""
    stmts = ast.parse(src).body
    CheckAndInlineConstants.run(stmts, __file__)
    assert ast.unparse(stmts).endswith("y = 8")


def test_const_correctly_evaluated_IV():
    src = """
a: Const[i32] = 4
def foo(y: i32):
    x: i32 = a + y
"""
    stmts = ast.parse(src).body
    CheckAndInlineConstants.run(stmts, __file__)
    assert ast.unparse(stmts).endswith("x: i32 = 4 + y")


def test_const_correctly_evaluated_V():
    src = """
a: Const[i32] = 4
b: Const[i32] = 4
def foo(y: i32):
    c: Const[i32] = a + b + 2
    x: i32 = c
"""
    stmts = ast.parse(src).body
    CheckAndInlineConstants.run(stmts, __file__)
    assert ast.unparse(stmts).endswith("x: i32 = 10")


def test_raises_exception_on_assignemnt_to_const_I():
    src = """
a: Const[i32] = 2 ** 5
a = 34
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert err.value.msg == "Constant 'a' is already defined and cannot be assigned to."


def test_raises_exception_on_assignemnt_to_const_II():
    src = """
x: Const[i32] = 100
def foo():
    x: i32 = 2
    return
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert err.value.msg == "Constant 'x' is already defined."


def test_raises_exception_on_assignemnt_to_const_III():
    src = """
y: Const[i32] = 100
@block
def bb0():
    y = 32
    return
bb0()
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert err.value.msg == "Constant 'y' is already defined and cannot be assigned to."


def test_raises_exception_on_assignemnt_to_const_IV():
    src = """
z: Const[i32] = 100
def foo(x: i32):
    @block
    def bb0(z: i32):
        return
    bb0(x)
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert (
        err.value.msg
        == "Constant 'z' is already defined and cannot be used as a function/block argument name."
    )


def test_raises_exception_on_duplicate_const():
    src = """
z: Const[i32] = 100
z: Const[i32] = 2
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert err.value.msg == "Constant 'z' is already defined."


def test_raises_exception_on_evaluation_error_I():
    src = """
z: Const[i32] = 23 / 0
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert (
        err.value.msg
        == "Non-constant expression cannot be assigned to constant variable 'z' or cannot be evaluated."
    )


def test_raises_exception_on_evaluation_error_II():
    src = """
a: Const[i32] = x + 12
"""
    stmts = ast.parse(src).body
    with pytest.raises(CodeGenerationException) as err:
        CheckAndInlineConstants.run(stmts, __file__)
    assert (
        err.value.msg
        == "Non-constant expression cannot be assigned to constant variable 'a' or cannot be evaluated."
    )
