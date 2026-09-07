from collections.abc import Sequence

import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.ir import BlockArgument
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from xdsl.transforms.scf_for_loop_range_folding import ScfForLoopRangeFoldingPass

index = IndexType()


def build_sum_scaled_module(factor_value: int) -> ModuleOp:
    @ModuleOp
    @Builder.implicit_region
    def module_op():
        with ImplicitBuilder(func.FuncOp("sum_scaled", ((index,), (index,))).body) as (
            ub,
        ):
            lb = arith.ConstantOp.from_int_and_width(0, index)
            step = arith.ConstantOp.from_int_and_width(1, index)
            factor = arith.ConstantOp.from_int_and_width(factor_value, index)
            initial = arith.ConstantOp.from_int_and_width(0, index)

            @Builder.implicit_region((index, index))
            def for_loop_region(args: Sequence[BlockArgument]):
                i, acc = args
                scaled = arith.MuliOp(i, factor)
                total = arith.AddiOp(scaled, acc)
                scf.YieldOp(total)

            result = scf.ForOp(lb, ub, step, (initial,), for_loop_region)
            func.ReturnOp(result)

    return module_op


def run_sum_scaled(module_op: ModuleOp, ub: int) -> int:
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ScfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("sum_scaled", (ub,))
    return result


@pytest.mark.parametrize("ub", [0, 1, 2, 4, 5])
def test_positive_factor_folding_preserves_index_sum(ub: int):
    original = build_sum_scaled_module(2)
    original.verify()
    expected = run_sum_scaled(original, ub)

    transformed = original.clone()
    transformed.verify()
    ScfForLoopRangeFoldingPass().apply(Context(), transformed)
    transformed.verify()

    # The original loop has step 1 and the transformed loop has step 2; both
    # compute the same pure sum for every tested upper bound.
    assert expected == sum(2 * i for i in range(0, ub, 1))
    assert run_sum_scaled(transformed, ub) == expected


@pytest.mark.parametrize("factor_value", [0, -2])
def test_nonpositive_factor_does_not_change_loop_range(factor_value: int):
    module = build_sum_scaled_module(factor_value)
    module.verify()

    transformed = module.clone()
    transformed.verify()
    ScfForLoopRangeFoldingPass().apply(Context(), transformed)
    transformed.verify()

    loop = next(op for op in transformed.walk() if isinstance(op, scf.ForOp))
    assert const_evaluate_operand(loop.step) == 1
    body_mul = next(op for op in loop.body.walk() if isinstance(op, arith.MuliOp))
    assert body_mul.operands[0] is loop.body.block.args[0]
