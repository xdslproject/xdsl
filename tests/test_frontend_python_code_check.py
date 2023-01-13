import ast
import pytest

from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.python_code_check import Constant, ConstantVisitor


def test_const_correctly_evaluated():
    visitor = ConstantVisitor()
    src = \
"""
a: Const[i32] = 2 ** 5
b: Const[i32] = len([1, 2, 3])
"""
    visitor.visit(ast.parse(src))

    assert "a" in visitor.constants
    assert not visitor.constants["a"].shadowed
    assert visitor.constants["a"].value == 32

    assert "b" in visitor.constants
    assert not visitor.constants["b"].shadowed
    assert visitor.constants["b"].value == 3


def test_can_assign_to_shadowed_constants_I():
    visitor = ConstantVisitor()
    visitor.constants["a"] = Constant(1, True)
    visitor.constants["b"] = Constant(2, True)
    src = \
"""
a = 3
b = 43
"""
    visitor.visit(ast.parse(src))
    assert visitor.constants["a"].value == 1
    assert visitor.constants["a"].shadowed
    assert visitor.constants["b"].value == 2
    assert visitor.constants["b"].shadowed


def test_can_assign_to_shadowed_constants_II():
    visitor = ConstantVisitor()
    src = \
"""
y: Const[i32] = 2 ** 5
def foo():
    y: i32 = 0
    y = 125
    return
"""
    visitor.visit(ast.parse(src))
    assert visitor.constants["y"].value == 32
    assert not visitor.constants["y"].shadowed


def test_raises_exception_on_assignemnt_to_const_I():
    visitor = ConstantVisitor()
    src = \
"""
a: Const[i32] = 2 ** 5
a = 34
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Cannot assign to constant variable 'a'."


def test_raises_exception_on_assignemnt_to_const_II():
    visitor = ConstantVisitor()
    src = \
"""
x: Const[i32] = 100
def foo():
    x = 2
    return
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Cannot assign to constant variable 'x'."


def test_raises_exception_on_assignemnt_to_const_III():
    visitor = ConstantVisitor()
    src = \
"""
y: Const[i32] = 100
@block
def bb0():
    y = 32
    return
bb0()
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Cannot assign to constant variable 'y'."


def test_raises_exception_on_assignemnt_to_const_IV():
    visitor = ConstantVisitor()
    src = \
"""
z: Const[i32] = 100
def foo():
    @block
    def bb0():
        z = 32
        return
    bb0()
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Cannot assign to constant variable 'z'."


def test_raises_exception_on_duplicate_const():
    visitor = ConstantVisitor()
    src = \
"""
z: Const[i32] = 100
z: Const[i32] = 2
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Constant 'z' is already defined in the program."


def test_raises_exception_on_evaluation_error_I():
    visitor = ConstantVisitor()
    src = \
"""
z: Const[i32] = 23 / 0
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Non-constant expression cannot be assigned to constant variable 'z' or cannot be evaluated."


def test_raises_exception_on_evaluation_error_II():
    visitor = ConstantVisitor()
    src = \
"""
a: Const[i32] = 12
b: Const[i32] = a + 23
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "Non-constant expression cannot be assigned to constant variable 'b' or cannot be evaluated."

def test_raises_exception_on_const_inside_functions():
    visitor = ConstantVisitor()
    src = \
"""
def foo():
    a: Const[i32] = 12
"""
    with pytest.raises(CodeGenerationException) as err:
        visitor.visit(ast.parse(src))
    assert err.value.msg == "All constant expressions have to be created in the global scope."
