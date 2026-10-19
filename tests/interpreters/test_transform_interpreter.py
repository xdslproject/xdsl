import re
from unittest.mock import patch

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import OperationHandle, TransformFunctions
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.transforms import get_all_passes
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


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


def test_apply_registered_pass():
    ctx = Context()

    op = transform.ApplyRegisteredPassOp(
        "canonicalize", create_ssa_value(transform.AnyOpType())
    )
    module_payload = builtin.ModuleOp([op])
    interpreter = Interpreter(module_payload)
    interpreter.register_implementations(
        TransformFunctions(ctx, {"canonicalize": lambda: CanonicalizePass})
    )

    module_payload = builtin.ModuleOp([])
    with patch.object(CanonicalizePass, "apply", autospec=True) as apply_0:
        (result,) = interpreter.run_op(op, (OperationHandle(module_payload),))

    assert result.ops == (module_payload,)
    apply_0.assert_called_once_with(CanonicalizePass(), ctx, module_payload)

    constant_payload = arith.ConstantOp(builtin.IntegerAttr(1, builtin.i32))
    with patch.object(CanonicalizePass, "apply", autospec=True) as apply_1:
        with pytest.raises(
            InterpretationError,
            match=re.escape(
                "transform.apply_registered_pass currently supports only "
                "builtin.module targets"
            ),
        ):
            interpreter.run_op(op, (OperationHandle(constant_payload),))

    apply_1.assert_not_called()

    with patch.object(CanonicalizePass, "apply", autospec=True) as apply_2:
        with pytest.raises(
            InterpretationError,
            match=re.escape(
                "transform.apply_registered_pass currently supports only "
                "builtin.module targets"
            ),
        ):
            interpreter.run_op(op, (OperationHandle(module_payload, constant_payload),))

    apply_2.assert_not_called()
