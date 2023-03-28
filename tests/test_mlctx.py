import pytest
from xdsl.dialects.builtin import UnregisteredOp

from xdsl.ir import MLContext, ParametrizedAttribute
from xdsl.irdl import irdl_op_definition, irdl_attr_definition, Operation


@irdl_op_definition
class DummyOp(Operation):
    name = "dummy"


@irdl_op_definition
class DummyOp2(Operation):
    name = "dummy2"


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = "dummy_attr"


@irdl_attr_definition
class DummyAttr2(ParametrizedAttribute):
    name = "dummy_attr2"


def test_get_op():
    """Test `get_op` and `get_optional_op` methods."""
    ctx = MLContext()
    ctx.register_op(DummyOp)

    assert ctx.get_op("dummy") == DummyOp
    with pytest.raises(Exception):
        _ = ctx.get_op("dummy2")

    assert ctx.get_optional_op("dummy") == DummyOp
    assert ctx.get_optional_op("dummy2") == None


def test_get_op_unregistered():
    """
    Test `get_op` and `get_optional_op`
    methods with the `unregistered_ops` flag.
    """
    ctx = MLContext()
    ctx.register_op(DummyOp)

    assert ctx.get_optional_op("dummy", allow_unregistered=True) == DummyOp
    assert isinstance(ctx.get_optional_op("dummy2", allow_unregistered=True),
                      UnregisteredOp)

    assert ctx.get_op("dummy", allow_unregistered=True) == DummyOp
    assert isinstance(ctx.get_op("dummy2", allow_unregistered=True),
                      UnregisteredOp)


def test_get_attr():
    """Test `get_attr` and `get_optional_attr` methods."""
    ctx = MLContext()
    ctx.register_attr(DummyAttr)

    assert ctx.get_attr("dummy_attr") == DummyAttr
    with pytest.raises(Exception):
        _ = ctx.get_attr("dummy_attr2")

    assert ctx.get_optional_attr("dummy_attr") == DummyAttr
    assert ctx.get_optional_attr("dummy_attr2") == None
