# RUN: python %s | filecheck %s

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import SSAValue


def add(operand1: int, operand2: int) -> int: ...


def pair(operand1: int, operand2: int) -> tuple[int, bool]: ...


def triple(operand1: int, operand2: int) -> tuple[int, int, int]: ...


def split(operand1: int, operand2: int) -> tuple[int, int]: ...


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
# Registered with fewer results than the functions return, to test unpacking them.
ctx.register_function(triple, arith.AddUIExtendedOp)
ctx.register_function(split, arith.AddiOp)
ctx.register_function(effect, build_effect)
ctx.register_function(consume, build_consume)


@ctx.parse_program
def assign_and_use(x: int) -> int:
    y = add(x, x)
    return y


print(assign_and_use.module)
# CHECK:       func.func @assign_and_use(%x: i32) -> i32 {
# CHECK-NEXT:    %0 = arith.addi %x, %x : i32
# CHECK-NEXT:    func.return %0 : i32
# CHECK-NEXT:  }


@ctx.parse_program
def reassign(x: int) -> int:
    y = add(x, x)
    y = add(y, x)
    x = add(x, y)
    return x


print(reassign.module)
# CHECK:       func.func @reassign(%x: i32) -> i32 {
# CHECK-NEXT:    %0 = arith.addi %x, %x : i32
# CHECK-NEXT:    %1 = arith.addi %0, %x : i32
# CHECK-NEXT:    %2 = arith.addi %x, %1 : i32
# CHECK-NEXT:    func.return %2 : i32
# CHECK-NEXT:  }


# The call is kept even though its result is never read.
@ctx.parse_program
def unused(x: int) -> None:
    _result = effect(x)


print(unused.module)
# CHECK:       func.func @unused(%x: i32) {
# CHECK-NEXT:    %0 = "test.op"(%x) : (i32) -> i32
# CHECK-NEXT:    func.return
# CHECK-NEXT:  }


@ctx.parse_program
def chained(x: int) -> int:
    y = _z = x
    return y


try:
    chained.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Only assignment to a variable or a tuple of variables is supported.


@ctx.parse_program
def unpacked(x: int) -> int:
    y, _z = pair(x, x)
    return y


print(unpacked.module)
# CHECK:       func.func @unpacked(%x: i32) -> i32 {
# CHECK-NEXT:    %0, %1 = arith.addui_extended %x, %x : i32, i1
# CHECK-NEXT:    func.return %0 : i32
# CHECK-NEXT:  }


@ctx.parse_program
def swapped(x: int, y: int) -> int:
    x, y = y, x
    return add(x, add(x, y))


print(swapped.module)
# CHECK:       func.func @swapped(%x: i32, %y: i32) -> i32 {
# CHECK-NEXT:    %0 = arith.addi %y, %x : i32
# CHECK-NEXT:    %1 = arith.addi %y, %0 : i32
# CHECK-NEXT:    func.return %1 : i32
# CHECK-NEXT:  }


@ctx.parse_program
def too_many(x: int) -> int:
    y, _z, _w = triple(x, x)
    return y


try:
    too_many.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Expected 3 values to unpack, got 2.


@ctx.parse_program
def single(x: int) -> int:
    y, _z = split(x, x)
    return y


try:
    single.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Expected 2 values to unpack, got 1.


@ctx.parse_program
def kept_tuple(x: int) -> int:
    _y = (x, x)
    return x


try:
    kept_tuple.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Tuples are only supported when unpacked.


@ctx.parse_program
def nested(x: int) -> int:
    (y, _z), _w = (x, x), x
    return y


try:
    nested.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Only assignment to a variable or a tuple of variables is supported.


@ctx.parse_program
def changed_type(x: int, flag: bool) -> int:
    y = x
    y = flag
    return y


try:
    changed_type.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Cannot change the type of variable 'y'.


@ctx.parse_program
def no_value(x: int) -> None:
    _result = consume(x)


try:
    no_value.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Expected an expression with exactly one result.
