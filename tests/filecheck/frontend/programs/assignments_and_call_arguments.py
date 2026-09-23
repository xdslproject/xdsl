# RUN: python %s | filecheck %s

import ast
from typing import Any, ClassVar

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.code_generation import CodeGenerationVisitor
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.program import PyASTProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import SSAValue


def add(operand1: int, operand2: int) -> int:
    return operand1 + operand2


def pair(lhs: int, rhs: int) -> tuple[int, bool]:
    return lhs + rhs, False


def consume(value: int) -> None:
    pass


def build_consume(value: SSAValue) -> test.TestOp:
    return test.TestOp([value])


class StackCheckingVisitor(CodeGenerationVisitor):
    """Check expression statements, including calls with zero or multiple results."""

    checked: ClassVar[int] = 0

    def visit_Expr(self, node: ast.Expr) -> None:
        size = len(self.inserter.stack)
        super().visit_Expr(node)
        assert len(self.inserter.stack) == size
        StackCheckingVisitor.checked += 1


ctx = PyASTContext(code_generation_visitor=StackCheckingVisitor)
ctx.register_type(int, builtin.i32)
ctx.register_function(add, arith.AddiOp)
ctx.register_function(pair, arith.AddUIExtendedOp)
ctx.register_function(consume, build_consume)
ctx.register_literal(int, lambda value: arith.ConstantOp.from_int_and_width(value, 32))
ctx.register_literal(
    float, lambda value: arith.ConstantOp(builtin.FloatAttr(value, builtin.f64))
)


# Assignment binds a call result, reassignment uses the previous value, and both
# positional and keyword arguments can contain literals or nested calls.
@ctx.parse_program
def assignments(x: int) -> int:
    """A docstring is metadata, not a string constant to lower."""
    y = add(x, 1)
    y = add(y, 2)
    return add(add(y, 3), operand2=4)


assert assignments(5) == 15
print(assignments.module)


# Discarding a call's values must retain the operation, including its effects.
# The customized visitor also verifies that no results leak onto the stack.
@ctx.parse_program
def discarded(x: int) -> int:
    pair(x, 1)
    consume(add(x, 2))
    return add(x, 3)


print(discarded.module)
assert StackCheckingVisitor.checked == 2


@ctx.parse_program
def docstring_only() -> None:
    """A function containing only a docstring still needs a return."""


print(docstring_only.module)


# Exact parameterized annotations can be registered without erasing their type
# arguments. Reverse lookup uses the origin class for Python operator lookup.
alias_ctx = PyASTContext()
alias_type = builtin.TupleType((builtin.i32,))
alias_ctx.register_type(list[int], alias_type)
assert alias_ctx.type_registry.get_annotation(alias_type) is list


@alias_ctx.parse_program
def parameterized(x: list[int]) -> list[int]:
    return x


print(parameterized.module)


# CHECK:       builtin.module {
# CHECK-NEXT:    func.func @assignments(%x: i32) -> i32 {
# CHECK-NEXT:      %0 = arith.constant 1 : i32
# CHECK-NEXT:      %1 = arith.addi %x, %0 : i32
# CHECK-NEXT:      %2 = arith.constant 2 : i32
# CHECK-NEXT:      %3 = arith.addi %1, %2 : i32
# CHECK-NEXT:      %4 = arith.constant 3 : i32
# CHECK-NEXT:      %5 = arith.addi %3, %4 : i32
# CHECK-NEXT:      %6 = arith.constant 4 : i32
# CHECK-NEXT:      %7 = arith.addi %5, %6 : i32
# CHECK-NEXT:      func.return %7 : i32
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    func.func @discarded(%x: i32) -> i32 {
# CHECK-NEXT:      %0 = arith.constant 1 : i32
# CHECK-NEXT:      %1, %2 = arith.addui_extended %x, %0 : i32, i1
# CHECK-NEXT:      %3 = arith.constant 2 : i32
# CHECK-NEXT:      %4 = arith.addi %x, %3 : i32
# CHECK-NEXT:      "test.op"(%4) : (i32) -> ()
# CHECK-NEXT:      %5 = arith.constant 3 : i32
# CHECK-NEXT:      %6 = arith.addi %x, %5 : i32
# CHECK-NEXT:      func.return %6 : i32
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    func.func @docstring_only() {
# CHECK-NEXT:      func.return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    func.func @parameterized(%x: tuple<i32>) -> tuple<i32> {
# CHECK-NEXT:      func.return %x : tuple<i32>
# CHECK-NEXT:    }
# CHECK-NEXT:  }


def print_error(program: PyASTProgram[..., Any]) -> None:
    try:
        program.module
    except CodeGenerationException as error:
        print(error.msg)
    else:
        raise AssertionError("expected a frontend diagnostic")


@ctx.parse_program
def chained():
    _x = _y = 1


print_error(chained)
# CHECK: Only assignment to a single variable is supported.


@ctx.parse_program
def unpacked():
    _x, _y = pair(1, 2)


print_error(unpacked)
# CHECK: Only assignment to a single variable is supported.


@ctx.parse_program
def changed_type():
    x = 1
    x = 2.0
    return x


print_error(changed_type)
# CHECK: Cannot change the type of variable 'x'.


@ctx.parse_program
def missing_result():
    x = consume(1)
    return x


print_error(missing_result)
# CHECK: Expected an expression with exactly one result.


@ctx.parse_program
def multiple_results():
    x = pair(1, 2)
    return x


print_error(multiple_results)
# CHECK: Expected an expression with exactly one result.


@ctx.parse_program
def multiple_results_argument():
    add(pair(1, 2), 3)  # pyright: ignore[reportArgumentType]


print_error(multiple_results_argument)
# CHECK: Expected an expression with exactly one result.


@ctx.parse_program
def unpack_keywords():
    add(**{"operand1": 1, "operand2": 2})


print_error(unpack_keywords)
# CHECK: Unpacking keyword arguments is not supported.


@ctx.parse_program
def read_before_assignment() -> int:
    x = add(x, 1)  # noqa: F821  # pyright: ignore[reportUnboundVariable, reportUnknownArgumentType]
    return x


print_error(read_before_assignment)
# CHECK: Symbol 'x' is not defined.


@ctx.parse_program
def ordinary_string():
    """Skip only the docstring."""
    "not a docstring"


print_error(ordinary_string)
# CHECK: Unsupported constant 'not a docstring' of type 'str'.
