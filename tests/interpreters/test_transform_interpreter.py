from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import TransformFunctions
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.transforms import get_all_passes


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
        transform.YieldOp(block.args[0])

    module = builtin.ModuleOp(
        [], attributes={"transform.with_named_sequence": builtin.UnitAttr()}
    )
    with ImplicitBuilder(module.body):
        body = Region(block)
        sym_name = "__transform_main"
        function_type = builtin.FunctionType.from_lists([ty], [ty])
        named_sequence = transform.NamedSequenceOp(sym_name, function_type, body)

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(transform.Transform)

    interpreter = Interpreter(module)
    interpreter.register_implementations(TransformFunctions(ctx, get_all_passes()))

    expected = Parser(ctx, payload).parse_module()
    (observed,) = interpreter.call_op(named_sequence, (expected,))
    assert expected is observed
