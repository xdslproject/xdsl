import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, polynomial, test
from xdsl.dialects.builtin import Builtin, ModuleOp, f32
from xdsl.transforms.polynomial_eval_to_arith import PolynomialEvalToArithPass
from xdsl.utils.exceptions import PassFailedException


@pytest.mark.parametrize("scheme", list(polynomial.EvalScheme))
def test_every_scheme_has_lowering(scheme: polynomial.EvalScheme):
    """
    This tests if every member of `EvalScheme` in the polynomial dialect has a defined lowering to arithmetic ops.
    """
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(polynomial.Polynomial)
    ctx.load_dialect(test.Test)

    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        x = test.TestOp(result_types=[f32]).results[0]
        polynomial.EvalOp.get(
            value=x,
            coefficients=(1.0, 2.0, 3.0),
            element_type=f32,
            scheme=scheme,
        )

    try:
        PolynomialEvalToArithPass().apply(ctx, module)
    except PassFailedException as e:
        pytest.fail(
            f"EvalScheme.{scheme.name} has no dispatch branch in "
            f"polynomial-eval-to-arith. Add a case for it in "
            f"PolynomialEvalToArith.match_and_rewrite. ({e})"
        )
