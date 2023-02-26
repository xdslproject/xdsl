import pytest

from xdsl.ir import MLContext, ParametrizedAttribute
from xdsl.irdl import irdl_op_definition, irdl_attr_definition, Operation


class DummyOp(Operation):
    name = "dummy"


class DummyOp2(Operation):
    name = "dummy2"


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = "dummy_attr"


@irdl_attr_definition
class DummyAttr2(ParametrizedAttribute):
    name = "dummy_attr2"


def test_registration_exceptions():
    ctx = MLContext()
    ctx.register_op(DummyOp)
    with pytest.raises(Exception):
        ctx.register_op(DummyOp)

    ctx.register_attr(DummyAttr)
    with pytest.raises(Exception):
        ctx.register_attr(DummyAttr)


def test_get_exceptions():
    ctx = MLContext()
    ctx.register_op(DummyOp)
    ctx.register_attr(DummyAttr)

    _ = ctx.get_op("dummy")
    with pytest.raises(Exception):
        _ = ctx.get_op("dummy2")

    _ = ctx.get_optional_attr("dummy_attr")
    assert ctx.get_optional_attr("dummy_attr2") is None

    with pytest.raises(Exception):
        _ = ctx.get_attr("dummy_attr2")
