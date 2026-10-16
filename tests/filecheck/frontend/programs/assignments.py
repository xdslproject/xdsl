# RUN: python %s | filecheck %s

import pytest

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException


def add(operand1: int, operand2: int) -> int:
    return operand1 + operand2


ctx = PyASTContext()
ctx.register_type(int, builtin.i32)
ctx.register_type(float, builtin.f64)
ctx.register_function(add, arith.AddiOp)
ctx.register_literal(int, lambda value: arith.ConstantOp.from_int_and_width(value, 32))


def assignments(x: int) -> int:
    y = add(x, 1)
    add(x, 2)
    y = add(y, 3)
    return add(y, operand2=x)


assert assignments(5) == 14
# Parsing an existing function must also work without decorator syntax.
print(ctx.parse_program(assignments).module)
# CHECK:      func.func @assignments(%x: i32) -> i32 {
# CHECK-NEXT:   %0 = arith.constant 1 : i32
# CHECK-NEXT:   %1 = arith.addi %x, %0 : i32
# CHECK-NEXT:   %2 = arith.constant 2 : i32
# CHECK-NEXT:   %3 = arith.addi %x, %2 : i32
# CHECK-NEXT:   %4 = arith.constant 3 : i32
# CHECK-NEXT:   %5 = arith.addi %1, %4 : i32
# CHECK-NEXT:   %6 = arith.addi %5, %x : i32
# CHECK-NEXT:   func.return %6 : i32
# CHECK-NEXT: }


@ctx.parse_program
def chained() -> None:
    _x = _y = 1


@ctx.parse_program
def unpacked() -> None:
    _x, _y = (1, 2)


for program in (chained, unpacked):
    with pytest.raises(
        CodeGenerationException, match="assignment to a single variable"
    ):
        program.module


@ctx.parse_program
def changed_type(x: int, y: float) -> int:
    x = y  # pyright: ignore[reportAssignmentType]
    return x


with pytest.raises(CodeGenerationException, match="Cannot change the type"):
    changed_type.module
