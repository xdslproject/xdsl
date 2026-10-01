# RUN: python %s | filecheck %s

from collections.abc import Sequence

from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import HasParent, IsolatedFromAbove, IsTerminator, NoTerminator
from xdsl.utils.lexer import Location


@irdl_op_definition
class CustomFuncOp(IRDLOperation):
    name = "test.func"

    sym_name = prop_def(builtin.StringAttr)
    function_type = prop_def(builtin.FunctionType)
    body = region_def("single_block")

    traits = traits_def(IsolatedFromAbove(), NoTerminator())

    def __init__(self, name: str, signature: builtin.FunctionType, body: Region):
        super().__init__(
            properties={
                "sym_name": builtin.StringAttr(name),
                "function_type": signature,
            },
            regions=[body],
        )


@irdl_op_definition
class CustomReturnOp(IRDLOperation):
    name = "test.return"

    arguments = var_operand_def()

    traits = traits_def(IsTerminator(), HasParent(CustomFuncOp))

    def __init__(self, values: Sequence[SSAValue]):
        super().__init__(operands=[values])


ctx = PyASTContext()
ctx.register_type(int, builtin.i32)
ctx.register_function_definition(
    lambda name, signature, body, location: CustomFuncOp(name, signature, body)
)
ctx.register_return(lambda values, location: CustomReturnOp(values))


@ctx.parse_program
def identity(value: int) -> int:
    return value


print(identity.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   "test.func"() <{sym_name = "identity", function_type = (i32) -> i32}> ({
# CHECK-NEXT:   ^bb0(%value: i32):
# CHECK-NEXT:     "test.return"(%value) : (i32) -> ()
# CHECK-NEXT:   }) : () -> ()
# CHECK-NEXT: }


@ctx.parse_program
def empty() -> None:
    pass


print(empty.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   "test.func"() <{sym_name = "empty", function_type = () -> ()}> ({
# CHECK-NEXT:     "test.return"() : () -> ()
# CHECK-NEXT:   }) : () -> ()
# CHECK-NEXT: }


ctx.register_return(lambda values, location: None)


@ctx.parse_program
def omitted() -> None:
    return None


print(omitted.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   "test.func"() <{sym_name = "omitted", function_type = () -> ()}> ({
# CHECK-NEXT:   ^bb0:
# CHECK-NEXT:   }) : () -> ()
# CHECK-NEXT: }


# A constructor may nest the function body inside another operation.
ctx.register_function_definition(
    lambda name, signature, body, location: builtin.ModuleOp(
        [CustomFuncOp(name, signature, body)]
    )
)


@ctx.parse_program
def wrapped() -> None:
    return


print(wrapped.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module {
# CHECK-NEXT:     "test.func"() <{sym_name = "wrapped", function_type = () -> ()}> ({
# CHECK-NEXT:     ^bb0:
# CHECK-NEXT:     }) : () -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }


def reject_function(
    name: str, signature: builtin.FunctionType, body: Region, location: Location
) -> CustomFuncOp:
    raise CodeGenerationException(*location, f"Rejected function '{name}'.")


ctx.register_function_definition(reject_function)


@ctx.parse_program
def invalid_function() -> None:
    pass


try:
    invalid_function.module
except CodeGenerationException as error:
    print(error)
# CHECK: Code generation exception at "{{.*}}custom_function.py", line 2 column 1: Rejected function 'invalid_function'.


def reject_return(values: Sequence[SSAValue], location: Location) -> None:
    raise CodeGenerationException(*location, "Rejected return value.")


ctx.register_function_definition(
    lambda name, signature, body, location: CustomFuncOp(name, signature, body)
)
ctx.register_return(reject_return)


@ctx.parse_program
def invalid_return(value: int) -> int:
    return value


try:
    invalid_return.module
except CodeGenerationException as error:
    print(error)
# CHECK: Code generation exception at "{{.*}}custom_function.py", line 3 column 5: Rejected return value.
