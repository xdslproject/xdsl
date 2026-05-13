import pytest

from xdsl.context import Context
from xdsl.dialects import math as math_dialect
from xdsl.dialects import polynomial
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    i64,
)
from xdsl.transforms.lower_exp_to_polynomial import (
    OVERFLOW_UPPER_BOUND,
    UNDERFLOW_LOWER_BOUND,
    LowerExpToPolynomialPass,
)
from xdsl.utils.test_value import create_ssa_value

_FloatTy = Float16Type | Float32Type | Float64Type

PRECISIONS = pytest.mark.parametrize(
    "elem_ty", [Float16Type(), Float32Type(), Float64Type()]
)


def _run_pass(elem_ty: _FloatTy, attrs: dict[str, int | float]) -> ModuleOp:
    operand = create_ssa_value(elem_ty)
    exp_op = math_dialect.ExpOp(operand)
    for name, value in attrs.items():
        if name == "max_bits_lost":
            exp_op.attributes[name] = IntegerAttr(int(value), i64)
        else:
            exp_op.attributes[name] = FloatAttr(float(value), elem_ty)
    module = ModuleOp([exp_op])
    ctx = Context()
    ctx.load_op(ModuleOp)
    LowerExpToPolynomialPass().apply(ctx, module)
    return module


@PRECISIONS
def test_default_no_bounds(elem_ty: _FloatTy):
    """No max_bits_lost, no [lower, upper] -> lowered with default target
    (correctly-rounded) and default [underflow, overflow] domain."""
    module = _run_pass(elem_ty, {})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == UNDERFLOW_LOWER_BOUND[type(elem_ty)]
    assert eval_op.domain_upper.value.data == OVERFLOW_UPPER_BOUND[type(elem_ty)]


@PRECISIONS
def test_default_with_bounds(elem_ty: _FloatTy):
    """No max_bits_lost, [lower, upper] given -> lowered with default target
    and the user-supplied domain."""
    module = _run_pass(elem_ty, {"lower_bound": -0.5, "upper_bound": 0.5})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == -0.5
    assert eval_op.domain_upper.value.data == 0.5


@PRECISIONS
def test_max_bits_lost_only(elem_ty: _FloatTy):
    """max_bits_lost only -> default interval [underflow, overflow]."""
    module = _run_pass(elem_ty, {"max_bits_lost": 2})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == UNDERFLOW_LOWER_BOUND[type(elem_ty)]
    assert eval_op.domain_upper.value.data == OVERFLOW_UPPER_BOUND[type(elem_ty)]


@PRECISIONS
def test_max_bits_lost_bounds_in_range(elem_ty: _FloatTy):
    """max_bits_lost + bounds entirely in range -> bounds used as-is."""
    module = _run_pass(
        elem_ty,
        {"max_bits_lost": 2, "lower_bound": -0.5, "upper_bound": 0.5},
    )
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == -0.5
    assert eval_op.domain_upper.value.data == 0.5


@PRECISIONS
def test_max_bits_lost_lower_out_of_range(elem_ty: _FloatTy):
    """max_bits_lost + lower < underflow -> lower clamped to underflow.

    -1000 is below every supported precision's underflow (f16 ~= -16.64,
    bf16 ~= -92.18, f32 ~= -103.28, f64 ~= -744.44).
    """
    module = _run_pass(
        elem_ty,
        {"max_bits_lost": 2, "lower_bound": -1000.0, "upper_bound": 0.5},
    )
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == UNDERFLOW_LOWER_BOUND[type(elem_ty)]
    assert eval_op.domain_upper.value.data == 0.5


@PRECISIONS
def test_max_bits_lost_upper_out_of_range(elem_ty: _FloatTy):
    """max_bits_lost + upper > overflow -> upper clamped to overflow.

    1000 exceeds every supported precision's overflow (f16 ~= 11.09,
    f32/bf16 ~= 88.72, f64 ~= 709.78).
    """
    module = _run_pass(
        elem_ty,
        {"max_bits_lost": 2, "lower_bound": -0.5, "upper_bound": 1000.0},
    )
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == -0.5
    assert eval_op.domain_upper.value.data == OVERFLOW_UPPER_BOUND[type(elem_ty)]


@PRECISIONS
def test_max_bits_lost_lower_only_in_range(elem_ty: _FloatTy):
    """max_bits_lost + lower only (in range) -> [lower, overflow]."""
    module = _run_pass(elem_ty, {"max_bits_lost": 2, "lower_bound": -0.5})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == -0.5
    assert eval_op.domain_upper.value.data == OVERFLOW_UPPER_BOUND[type(elem_ty)]


@PRECISIONS
def test_max_bits_lost_lower_only_out_of_range(elem_ty: _FloatTy):
    """max_bits_lost + lower only (out of range) -> [underflow, overflow]."""
    module = _run_pass(elem_ty, {"max_bits_lost": 2, "lower_bound": -1000.0})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == UNDERFLOW_LOWER_BOUND[type(elem_ty)]
    assert eval_op.domain_upper.value.data == OVERFLOW_UPPER_BOUND[type(elem_ty)]


@PRECISIONS
def test_max_bits_lost_upper_only_in_range(elem_ty: _FloatTy):
    """max_bits_lost + upper only (in range) -> [underflow, upper]."""
    module = _run_pass(elem_ty, {"max_bits_lost": 2, "upper_bound": 0.5})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == UNDERFLOW_LOWER_BOUND[type(elem_ty)]
    assert eval_op.domain_upper.value.data == 0.5


@PRECISIONS
def test_max_bits_lost_upper_only_out_of_range(elem_ty: _FloatTy):
    """max_bits_lost + upper only (out of range) -> [underflow, overflow]."""
    module = _run_pass(elem_ty, {"max_bits_lost": 2, "upper_bound": 1000.0})
    eval_op = next(iter(module.body.block.ops))
    assert isinstance(eval_op, polynomial.EvalOp)
    assert eval_op.domain_lower is not None
    assert eval_op.domain_upper is not None
    assert eval_op.domain_lower.value.data == UNDERFLOW_LOWER_BOUND[type(elem_ty)]
    assert eval_op.domain_upper.value.data == OVERFLOW_UPPER_BOUND[type(elem_ty)]
