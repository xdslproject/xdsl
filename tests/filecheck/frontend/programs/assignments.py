# RUN: python %s | filecheck %s

from typing import Any

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.program import PyASTProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import SSAValue


def add(operand1: int, operand2: int) -> int: ...


def pair(operand1: int, operand2: int) -> tuple[int, bool]: ...


def effect(value: int) -> int: ...


def consume(value: int) -> None: ...


def build_effect(value: SSAValue) -> test.TestOp:
    return test.TestOp([value], result_types=[value.type])


def build_consume(value: SSAValue) -> test.TestOp:
    return test.TestOp([value])


ctx = PyASTContext()
ctx.register_type(int, builtin.i32)
ctx.register_type(bool, builtin.i1)
ctx.register_function(add, arith.AddiOp)
ctx.register_function(pair, arith.AddUIExtendedOp)
ctx.register_function(effect, build_effect)
ctx.register_function(consume, build_consume)


@ctx.parse_program
def assign_and_use(x: int) -> int:
    y = add(x, x)
    return y


@ctx.parse_program
def reassign(x: int) -> int:
    y = add(x, x)
    y = add(y, x)
    x = add(x, y)
    return x


# The call is kept even though its result is never read.
@ctx.parse_program
def unused(x: int) -> None:
    _result = effect(x)


@ctx.parse_program
def unpacked(x: int) -> bool:
    _y, z = pair(x, x)
    return z


@ctx.parse_program
def swapped(x: int, y: int) -> int:
    x, y = y, x
    return add(x, effect(y))


for program in (assign_and_use, reassign, unused, unpacked, swapped):
    print(program.module)

# CHECK:       func.func @assign_and_use(%x: i32) -> i32 {
# CHECK-NEXT:    %0 = arith.addi %x, %x : i32
# CHECK-NEXT:    func.return %0 : i32
# CHECK-NEXT:  }

# CHECK:       func.func @reassign(%x: i32) -> i32 {
# CHECK-NEXT:    %0 = arith.addi %x, %x : i32
# CHECK-NEXT:    %1 = arith.addi %0, %x : i32
# CHECK-NEXT:    %2 = arith.addi %x, %1 : i32
# CHECK-NEXT:    func.return %2 : i32
# CHECK-NEXT:  }

# CHECK:       func.func @unused(%x: i32) {
# CHECK-NEXT:    %0 = "test.op"(%x) : (i32) -> i32
# CHECK-NEXT:    func.return
# CHECK-NEXT:  }

# CHECK:       func.func @unpacked(%x: i32) -> i1 {
# CHECK-NEXT:    %0, %1 = arith.addui_extended %x, %x : i32, i1
# CHECK-NEXT:    func.return %1 : i1
# CHECK-NEXT:  }

# CHECK:       func.func @swapped(%x: i32, %y: i32) -> i32 {
# CHECK-NEXT:    %0 = "test.op"(%x) : (i32) -> i32
# CHECK-NEXT:    %1 = arith.addi %y, %0 : i32
# CHECK-NEXT:    func.return %1 : i32
# CHECK-NEXT:  }


def print_error(program: PyASTProgram[..., Any]) -> None:
    try:
        program.module
    except CodeGenerationException as error:
        print(f"{program.name}: {error.msg}")
    else:
        raise AssertionError(f"expected {program.name} to be rejected")


@ctx.parse_program
def chained(x: int) -> int:
    y = _z = x
    return y


@ctx.parse_program
def nested(x: int) -> int:
    (y, _z), _w = (x, x), x
    return y


@ctx.parse_program
def too_many(x: int) -> int:
    y, _z, _w = pair(x, x)  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    return y


@ctx.parse_program
def tuple_value(x: int) -> None:
    _y = x, x


@ctx.parse_program
def changed_type(x: int, flag: bool) -> int:
    y = x
    y = flag
    return y


@ctx.parse_program
def no_value(x: int) -> None:
    _result = consume(x)


@ctx.parse_program
def inside_if(x: int, flag: bool) -> int:
    y = x
    if flag:
        y = add(x, x)
    return y


for program in (
    chained,
    nested,
    too_many,
    tuple_value,
    changed_type,
    no_value,
    inside_if,
):
    print_error(program)

# CHECK:      chained: Only assignment to a variable or a tuple of variables is supported.
# CHECK-NEXT: nested: Only assignment to a variable or a tuple of variables is supported.
# CHECK-NEXT: too_many: Expected 3 values to unpack, but got 2.
# CHECK-NEXT: tuple_value: Expected an expression with exactly one result.
# CHECK-NEXT: changed_type: Cannot change the type of variable 'y'.
# CHECK-NEXT: no_value: Expected an expression with exactly one result.
# CHECK-NEXT: inside_if: Assignments are only supported directly in the function body.
