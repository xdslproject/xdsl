# RUN: python %s | filecheck %s

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException

ctx = PyASTContext()
ctx.register_type(int, bigint.bigint)
ctx.register_type(bool, builtin.i1)


@ctx.parse_program
def redefined_function():  # pyright: ignore[reportRedeclaration]
    return


@ctx.parse_program
def redefined_function(x: int) -> int:
    return x


# Function definitions can be overriden
print(redefined_function.module)
# CHECK:      builtin.module {
# CHECK-NEXT:     func.func @redefined_function(%x : !bigint.bigint) -> !bigint.bigint {
# CHECK-NEXT:       func.return %x : !bigint.bigint
# CHECK-NEXT:     }
# CHECK-NEXT:   }


# CHECK-NEXT: Cannot have an inner function 'inner' inside another function.
@ctx.parse_program
def outer():
    def inner():  # pyright: ignore[reportUnusedFunction]
        return

    return


try:
    outer.module
except FrontendProgramException as e:
    print(e.msg)


# CHECK-NEXT: Expected non-zero number of return types in function 'missing_return_value', but got 0.
@ctx.parse_program
def missing_return_value() -> int:
    return  # pyright: ignore[reportReturnType]


try:
    missing_return_value.module
except FrontendProgramException as e:
    print(e.msg)


try:
    # CHECK-NEXT: Cannot re-register type name 'int'
    ctx.register_type(int, bigint.bigint)
except FrontendProgramException as e:
    print(e.msg)


try:
    # CHECK-NEXT: Cannot re-register function 'int.__add__'
    ctx.register_function(int.__add__, bigint.AddOp)
    ctx.register_function(int.__add__, bigint.AddOp)
except FrontendProgramException as e:
    print(e.msg)


try:
    # CHECK-NEXT: Cannot register multiple source types for IR type '!bigint.bigint'
    ctx.register_type(float, bigint.bigint)
except FrontendProgramException as e:
    print(e.msg)


try:
    # CHECK-NEXT: Cannot register multiple source types for IR type '!bigint.bigint'
    ctx.register_type(float, bigint.bigint)
except FrontendProgramException as e:
    print(e.msg)


# CHECK: Expected non-zero number of return types in function 'test_no_return_type', but got 0.
@ctx.parse_program
def test_no_return_type(a: int) -> int:
    return  # pyright: ignore[reportReturnType]


try:
    test_no_return_type.module
except FrontendProgramException as e:
    print(e.msg)


# CHECK: Type signature and the type of the return value do not match at position 0: expected i1, got !bigint.bigint.
@ctx.parse_program
def test_wrong_return_type(a: bool, b: int) -> bool:
    return b  # pyright: ignore[reportReturnType]


try:
    test_wrong_return_type.module
except FrontendProgramException as e:
    print(e.msg)


# CHECK: Expected no return types in function 'test_no_return_types'.
@ctx.parse_program
def test_no_return_types(a: int):
    return a


try:
    test_no_return_types.module
except FrontendProgramException as e:
    print(e.msg)


# CHECK: Expected the same types for binary operation 'Add', but got !bigint.bigint and i1.
@ctx.parse_program
def bin_op_type_mismatch(a: int, b: bool) -> int:
    return a + b


try:
    bin_op_type_mismatch.module
except FrontendProgramException as e:
    print(e.msg)


# CHECK: Expected the same types for comparison operator 'Lt', but got !bigint.bigint and i1.
@ctx.parse_program
def cmp_op_type_mismatch(a: int, b: bool) -> bool:
    return a < b


try:
    cmp_op_type_mismatch.module
except FrontendProgramException as e:
    print(e.msg)
