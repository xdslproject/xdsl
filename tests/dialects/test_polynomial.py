import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    TensorType,
    VectorType,
    f16,
    f32,
    f64,
)
from xdsl.dialects.polynomial import (
    ChebyshevPolynomialAttr,
    EvalChebyshevOp,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

# --- ChebyshevPolynomialAttr construction ---


def test_attr_from_float_list():
    attr = ChebyshevPolynomialAttr([0.5, 1.2, 0.3])
    assert attr.degree == 2
    assert attr.coeff_values == [0.5, 1.2, 0.3]
    assert attr.lower == -1.0
    assert attr.upper == 1.0


def test_attr_from_float_attr_list():
    coeffs = [FloatAttr(0.5, f64), FloatAttr(1.2, f64)]
    attr = ChebyshevPolynomialAttr(coeffs, -2.0, 3.0)
    assert attr.degree == 1
    assert attr.coeff_values == [0.5, 1.2]
    assert attr.lower == -2.0
    assert attr.upper == 3.0


def test_attr_from_array_attr():
    arr = ArrayAttr([FloatAttr(1.0, f64), FloatAttr(2.0, f64)])
    attr = ChebyshevPolynomialAttr(arr)
    assert attr.degree == 1
    assert attr.coeff_values == [1.0, 2.0]


def test_attr_custom_domain():
    attr = ChebyshevPolynomialAttr([1.0, 2.0], domain_lower=-10.0, domain_upper=0.0)
    assert attr.lower == -10.0
    assert attr.upper == 0.0


def test_attr_default_domain():
    attr = ChebyshevPolynomialAttr([1.0, 2.0])
    assert attr.lower == -1.0
    assert attr.upper == 1.0


def test_attr_single_coefficient():
    attr = ChebyshevPolynomialAttr([42.0])
    assert attr.degree == 0
    assert attr.coeff_values == [42.0]


# --- EvalChebyshevOp construction ---


def test_eval_basic_construction():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, [0.5, 1.2, 0.3])
    assert op.value.type == f32
    assert op.result.type == f32
    assert op.degree == 2
    assert op.polynomial.coeff_values == [0.5, 1.2, 0.3]


def test_eval_pre_built_attr():
    attr = ChebyshevPolynomialAttr([1.0, 2.0, 3.0], -5.0, 5.0)
    x = create_ssa_value(f64)
    op = EvalChebyshevOp(x, attr)
    assert op.polynomial is attr
    assert op.polynomial.lower == -5.0
    assert op.polynomial.upper == 5.0


@pytest.mark.parametrize(
    "tp",
    [f16, f32, f64, VectorType(f32, [4]), TensorType(f64, [8])],
    ids=["f16", "f32", "f64", "vector<4xf32>", "tensor<8xf64>"],
)
def test_eval_type_polymorphism(tp):
    x = create_ssa_value(tp)
    op = EvalChebyshevOp(x, [1.0, 2.0])
    assert op.value.type == tp
    assert op.result.type == tp


def test_eval_domain_bounds_propagated():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, [1.0, 2.0], domain_lower=-10.0, domain_upper=0.0)
    assert op.polynomial.lower == -10.0
    assert op.polynomial.upper == 0.0


# --- Verification ---


def test_verify_invalid_domain_lower_ge_upper():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, [1.0, 2.0], domain_lower=1.0, domain_upper=-1.0)
    with pytest.raises(VerifyException, match="domain_lower.*strictly less"):
        op.verify()


def test_verify_invalid_domain_equal():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, [1.0, 2.0], domain_lower=0.0, domain_upper=0.0)
    with pytest.raises(VerifyException, match="domain_lower.*strictly less"):
        op.verify()


def test_verify_invalid_degree_zero():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, [1.0], domain_lower=-1.0, domain_upper=1.0)
    with pytest.raises(VerifyException, match="at least degree 1"):
        op.verify()


def test_verify_valid_op():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, [0.5, 1.2, 0.3], domain_lower=-1.0, domain_upper=1.0)
    op.verify()
