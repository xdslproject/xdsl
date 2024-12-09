from xdsl.dialects import eqsat, pdl
from xdsl.dialects.builtin import (
    IntegerType,
    StringAttr,
)
from xdsl.interpreters.eqsat_pdl import EqsatPDLMatcher


def test_match_type():
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.TypeOp()
    ssa_value = pdl_op.result
    ssa_value = eqsat.EClassOp(ssa_value).result
    xdsl_value = StringAttr("a")

    # New value
    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    assert not matcher.match_type(ssa_value, pdl_op, StringAttr("b"))
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_fixed_type():
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(32))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result
    ssa_value = eqsat.EClassOp(ssa_value).result

    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_not_match_fixed_type():
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(64))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result
    ssa_value = eqsat.EClassOp(ssa_value).result

    assert not matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {}
