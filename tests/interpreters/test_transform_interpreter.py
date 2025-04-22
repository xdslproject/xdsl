from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import TransformFunctions
from xdsl.ir import Block, Region
from xdsl.parser import Parser

interpreter = Interpreter(builtin.ModuleOp([]))
interpreter.register_implementations(TransformFunctions())


def test_empty_transform_module():
    payload = """
    module {
        func.func @foo() {
            func.return
        }
    }
    """

    ty = transform.OperationType("builtin.module")
    block = Block(arg_types=[ty])
    with ImplicitBuilder(block):
        transform.YieldOp()

    module = builtin.ModuleOp(
        [], attributes={"transform.with_named_sequence": builtin.UnitAttr()}
    )
    with ImplicitBuilder(module.body):
        body = Region(block)
        sym_name = "__transform_main"
        function_type = builtin.FunctionType.from_lists([ty], [])
        named_sequence = transform.NamedSequenceOp(sym_name, function_type, body)

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(transform.Transform)

    module = Parser(ctx, payload).parse_module()
    expected = Parser(ctx, payload).parse_module()
    interpreter.call_op(named_sequence, (module,))
    # No changes because named sequence just contains yield
    # Compare string as proxy for structure equality.
    assert str(expected) == str(module)
