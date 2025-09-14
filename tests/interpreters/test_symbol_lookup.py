from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions


@ModuleOp
@Builder.implicit_region
def outer_module():
    @ModuleOp
    @Builder.implicit_region
    def nested():
        with ImplicitBuilder(func.FuncOp("nested_func", ((), (i32,))).body):
            a = arith.ConstantOp.from_int_and_width(42, 32).result
            func.ReturnOp(a)

    nested.sym_name = StringAttr("nested")
    _call_op = func.CallOp(
        SymbolRefAttr("nested", ("nested_func",)),
        (),
        (i32,),
    )


def test_nested_symbol_lookup():
    interpreter = Interpreter(outer_module)
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    assert interpreter.call_op(SymbolRefAttr("nested", ("nested_func",)), ()) == (42,)
    call_op = outer_module.regions[0].block.last_op
    assert isinstance(call_op, func.CallOp)
    assert interpreter.run_op(call_op, ()) == (42,)
