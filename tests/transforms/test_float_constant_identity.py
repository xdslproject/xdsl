import math
import struct

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)


@pytest.mark.parametrize(
    "pass_type", [CommonSubexpressionElimination, CanonicalizePass]
)
@pytest.mark.parametrize(
    "op_type", [arith.AddfOp, arith.MulfOp, arith.MinimumfOp, arith.MaximumfOp]
)
@pytest.mark.parametrize("zeros", [(0.0, -0.0), (-0.0, 0.0), (0.0, 0.0), (-0.0, -0.0)])
def test_signed_zero_constant_identity(
    pass_type: type[ModulePass],
    op_type: type[arith.AddfOp | arith.MulfOp | arith.MinimumfOp | arith.MaximumfOp],
    zeros: tuple[float, float],
):
    """Constant uniquing must preserve observable zero signs without fast math."""
    block = Block(arg_types=[builtin.f64])
    constants = [
        arith.ConstantOp(builtin.FloatAttr(zero, builtin.f64)) for zero in zeros
    ]
    operations = [op_type(constant, block.args[0]) for constant in constants]
    block.add_ops([*constants, *operations, func.ReturnOp(*operations)])
    module = builtin.ModuleOp(
        [
            func.FuncOp(
                "test", ([builtin.f64], [builtin.f64, builtin.f64]), Region(block)
            )
        ]
    )
    module.verify()

    inputs = (-math.inf, -2.0, -0.0, 0.0, 2.0, math.inf, math.nan)

    def evaluate():
        interpreter = Interpreter(module)
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(FuncFunctions())
        # Python numeric equality hides signed zero; compare every non-NaN bit.
        return tuple(
            tuple(
                "nan" if math.isnan(value) else struct.pack(">d", value)
                for value in interpreter.call_op("test", (arg,))
            )
            for arg in inputs
        )

    expected = evaluate()
    ctx = Context()
    for dialect in (arith.Arith, builtin.Builtin, func.Func):
        ctx.load_dialect(dialect)
    pass_type().apply(ctx, module)
    module.verify()

    assert evaluate() == expected
    same_sign = math.copysign(1.0, zeros[0]) == math.copysign(1.0, zeros[1])
    # Canonicalization is a preservation control here; only CSE is required to
    # merge the same-sign constants in this fixture.
    assert sum(isinstance(op, arith.ConstantOp) for op in module.walk()) == (
        1 if same_sign and pass_type is CommonSubexpressionElimination else 2
    )
