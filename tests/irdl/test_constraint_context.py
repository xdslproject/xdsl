import re

import pytest

from xdsl.dialects.builtin import i32, i64
from xdsl.irdl import InferenceContext, VerificationContext
from xdsl.utils.exceptions import VerifyException


def test_inference_context_empty():
    ctx = InferenceContext()
    assert not ctx.attr_variables
    assert not ctx.range_variables
    assert not ctx.int_variables

    assert ctx.get_variable("test") is None
    assert ctx.get_range_variable("test") is None
    assert ctx.get_int_variable("test") is None


def test_verification_context_empty():
    ctx = VerificationContext()
    assert not ctx.attr_variables
    assert not ctx.range_variables
    assert not ctx.int_variables

    assert ctx.get_variable("test") is None
    assert ctx.get_range_variable("test") is None
    assert ctx.get_int_variable("test") is None


def test_attr_roundtrip():
    ctx = VerificationContext()

    assert ctx.set_attr_variable("test", i32)

    assert ctx.get_variable("test") == i32

    assert ctx.get_range_variable("test") is None
    assert ctx.get_int_variable("test") is None
    assert ctx.get_variable("test2") is None

    assert len(ctx.attr_variables) == 1
    assert not ctx.range_variables
    assert not ctx.int_variables


def test_range_roundtrip():
    ctx = VerificationContext()

    assert ctx.set_range_variable("test", (i32, i32))

    assert ctx.get_range_variable("test") == (i32, i32)

    assert ctx.get_variable("test") is None
    assert ctx.get_int_variable("test") is None
    assert ctx.get_range_variable("test2") is None

    assert not ctx.attr_variables
    assert len(ctx.range_variables) == 1
    assert not ctx.int_variables


def test_int_roundtrip():
    ctx = VerificationContext()

    assert ctx.set_int_variable("test", 10)

    assert ctx.get_int_variable("test") == 10

    assert ctx.get_variable("test") is None
    assert ctx.get_range_variable("test") is None
    assert ctx.get_int_variable("test2") is None

    assert not ctx.attr_variables
    assert not ctx.range_variables
    assert len(ctx.int_variables) == 1


def test_attr_shadowing():
    ctx = VerificationContext()

    assert ctx.set_attr_variable("test", i32)
    assert not ctx.set_attr_variable("test", i32)


def test_range_shadowing():
    ctx = VerificationContext()

    assert ctx.set_range_variable("test", (i32, i32))
    assert not ctx.set_range_variable("test", (i32, i32))


def test_int_shadowing():
    ctx = VerificationContext()

    assert ctx.set_int_variable("test", 10)
    assert not ctx.set_int_variable("test", 10)


def test_attr_clash():
    ctx = VerificationContext()

    assert ctx.set_attr_variable("test", i32)

    with pytest.raises(
        VerifyException,
        match="attribute i32 expected from variable 'test', but got i64",
    ):
        ctx.set_attr_variable("test", i64)


def test_range_clash():
    ctx = VerificationContext()

    assert ctx.set_range_variable("test", (i32, i32))

    with pytest.raises(
        VerifyException,
        match=re.escape(
            "attributes ('i32', 'i32') expected from range variable 'test', but got ('i32', 'i64')"
        ),
    ):
        ctx.set_range_variable("test", (i32, i64))


def test_int_clash():
    ctx = VerificationContext()

    assert ctx.set_int_variable("test", 10)

    with pytest.raises(
        VerifyException, match="integer 10 expected from int variable 'test', but got 9"
    ):
        ctx.set_int_variable("test", 9)
