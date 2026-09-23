from dataclasses import dataclass

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import OperationHandle, TransformFunctions
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.transforms import get_all_passes


@pytest.mark.parametrize("num_targets", [0, 1, 2])
def test_empty_transform_module(num_targets: int):
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

    expected = OperationHandle(
        *(Parser(ctx, payload).parse_module() for _ in range(num_targets))
    )
    (observed,) = interpreter.call_op(named_sequence, (expected,))
    assert expected is observed


@pytest.mark.parametrize("num_targets", [0, 1, 2])
def test_apply_registered_pass(num_targets: int):
    visited: list[builtin.ModuleOp] = []

    @dataclass(frozen=True)
    class RecordPass(ModulePass):
        name = "record"

        def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
            visited.append(op)

    targets = OperationHandle(*(builtin.ModuleOp([]) for _ in range(num_targets)))
    block = Block(arg_types=[transform.AnyOpType()])
    op = transform.ApplyRegisteredPassOp("record", block.args[0])
    module = builtin.ModuleOp([op])
    interpreter = Interpreter(module)
    interpreter.register_implementations(
        TransformFunctions(Context(), {"record": lambda: RecordPass})
    )

    (result,) = interpreter.run_op(op, (targets,))

    assert result is targets
    assert visited == list(targets.ops)
