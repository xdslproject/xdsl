# RUN: python %s | filecheck %s

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.context import PyASTContext
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
def inside_if(x: int, flag: bool) -> int:
    y = x
    if flag:
        y = add(x, x)
    return y


print(inside_if.module)
# CHECK:       func.func @inside_if(%x: i32, %flag: i1) -> i32 {
# CHECK-NEXT:    %0 = scf.if %flag -> (i32) {
# CHECK-NEXT:      %1 = arith.addi %x, %x : i32
# CHECK-NEXT:      scf.yield %1 : i32
# CHECK-NEXT:    } else {
# CHECK-NEXT:      scf.yield %x : i32
# CHECK-NEXT:    }
# CHECK-NEXT:    func.return %0 : i32
# CHECK-NEXT:  }


# Variables first assigned in a branch can be used in that branch.
@ctx.parse_program
def both_branches(x: int, flag: bool) -> int:
    y = x
    if flag:
        y = add(y, x)
    else:
        z = add(x, x)
        y = add(z, z)
    return y


print(both_branches.module)
# CHECK:       func.func @both_branches(%x: i32, %flag: i1) -> i32 {
# CHECK-NEXT:    %0 = scf.if %flag -> (i32) {
# CHECK-NEXT:      %1 = arith.addi %x, %x : i32
# CHECK-NEXT:      scf.yield %1 : i32
# CHECK-NEXT:    } else {
# CHECK-NEXT:      %2 = arith.addi %x, %x : i32
# CHECK-NEXT:      %3 = arith.addi %2, %2 : i32
# CHECK-NEXT:      scf.yield %3 : i32
# CHECK-NEXT:    }
# CHECK-NEXT:    func.return %0 : i32
# CHECK-NEXT:  }


@ctx.parse_program
def elif_chain(x: int, a: bool, b: bool) -> int:
    y = x
    if a:
        y = add(y, y)
    elif b:
        x = add(x, y)
    return add(x, y)


print(elif_chain.module)
# CHECK:       func.func @elif_chain(%x: i32, %a: i1, %b: i1) -> i32 {
# CHECK-NEXT:    %0, %1 = scf.if %a -> (i32, i32) {
# CHECK-NEXT:      %2 = arith.addi %x, %x : i32
# CHECK-NEXT:      scf.yield %2, %x : i32, i32
# CHECK-NEXT:    } else {
# CHECK-NEXT:      %3 = scf.if %b -> (i32) {
# CHECK-NEXT:        %4 = arith.addi %x, %x : i32
# CHECK-NEXT:        scf.yield %4 : i32
# CHECK-NEXT:      } else {
# CHECK-NEXT:        scf.yield %x : i32
# CHECK-NEXT:      }
# CHECK-NEXT:      scf.yield %x, %3 : i32, i32
# CHECK-NEXT:    }
# CHECK-NEXT:    %5 = arith.addi %1, %0 : i32
# CHECK-NEXT:    func.return %5 : i32
# CHECK-NEXT:  }


@ctx.parse_program
def read_inside_if(x: int, flag: bool) -> int:
    return x if flag else add(x, x)


print(read_inside_if.module)
# CHECK:       func.func @read_inside_if(%x: i32, %flag: i1) -> i32 {
# CHECK-NEXT:    %0 = scf.if %flag -> (i32) {
# CHECK-NEXT:      scf.yield %x : i32
# CHECK-NEXT:    } else {
# CHECK-NEXT:      %1 = arith.addi %x, %x : i32
# CHECK-NEXT:      scf.yield %1 : i32
# CHECK-NEXT:    }
# CHECK-NEXT:    func.return %0 : i32
# CHECK-NEXT:  }


@ctx.parse_program
def chained(x: int) -> int:
    y = _z = x
    return y


try:
    chained.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Only assignment to a single variable is supported.


@ctx.parse_program
def unpacked(x: int) -> int:
    y, _z = pair(x, x)
    return y


try:
    unpacked.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Only assignment to a single variable is supported.


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


# The variable has no value when the branch is not taken.
@ctx.parse_program
def only_in_branch(x: int, flag: bool) -> int:
    if flag:
        y = add(x, x)
    return y  # pyright: ignore[reportPossiblyUnboundVariable]


try:
    only_in_branch.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Symbol 'y' is not defined.
