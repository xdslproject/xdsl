import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import UnregisteredAttr, UnregisteredOp
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition
from xdsl.utils.exceptions import (
    AlreadyRegisteredConstructException,
    UnregisteredConstructException,
)


@irdl_op_definition
class DummyOp(IRDLOperation):
    name = "test.dummy"


@irdl_op_definition
class DummyOp2(IRDLOperation):
    name = "test.dummy2"


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = "test.dummy_attr"


@irdl_attr_definition
class DummyAttr2(ParametrizedAttribute):
    name = "test.dummy_attr2"


@irdl_attr_definition
class DummyType(ParametrizedAttribute, TypeAttribute):
    name = "test.dummy_type"


testDialect = Dialect("test", [DummyOp], [DummyAttr, DummyType])
testDialect2 = Dialect("test", [DummyOp2], [DummyAttr2, DummyType])


def test_get_op():
    """Test `get_op` and `get_optional_op` methods."""
    ctx = Context()
    ctx.load_op(DummyOp)

    assert ctx.get_op("test.dummy") == DummyOp
    with pytest.raises(UnregisteredConstructException):
        _ = ctx.get_op("test.dummy2")

    assert ctx.get_optional_op("test.dummy") == DummyOp
    assert ctx.get_optional_op("test.dummy2") is None


def test_get_op_unregistered():
    """
    Test `get_op` and `get_optional_op`
    methods with the `allow_unregistered` flag.
    """
    ctx = Context(allow_unregistered=True)
    ctx.load_op(DummyOp)

    assert ctx.get_optional_op("test.dummy") == DummyOp
    op = ctx.get_optional_op("test.dummy2")
    assert op is not None
    assert issubclass(op, UnregisteredOp)

    assert ctx.get_op("test.dummy") == DummyOp
    assert issubclass(ctx.get_op("test.dummy2"), UnregisteredOp)


def test_get_op_with_dialect_stack():
    """Test `get_op` and `get_optional_op` methods."""
    ctx = Context()
    ctx.load_op(DummyOp)

    assert ctx.get_op("dummy", dialect_stack=("test",)) == DummyOp
    with pytest.raises(UnregisteredConstructException):
        _ = ctx.get_op("dummy2", dialect_stack=("test",))

    assert ctx.get_optional_op("dummy", dialect_stack=("test",)) == DummyOp
    assert ctx.get_optional_op("dummy2", dialect_stack=("test",)) is None


def test_get_op_unregistered_with_dialect_stack():
    """
    Test `get_op` and `get_optional_op`
    methods with the `allow_unregistered` flag.
    """
    ctx = Context(allow_unregistered=True)
    ctx.load_op(DummyOp)

    assert ctx.get_optional_op("dummy", dialect_stack=("test",)) == DummyOp
    op_type = ctx.get_optional_op("dummy2", dialect_stack=("test",))
    assert op_type is not None
    assert issubclass(op_type, UnregisteredOp)
    assert op_type.create().op_name.data == "dummy2"

    assert ctx.get_op("dummy", dialect_stack=("test",)) == DummyOp
    op_type = ctx.get_op("dummy2", dialect_stack=("test",))
    assert issubclass(op_type, UnregisteredOp)
    assert op_type.create().op_name.data == "dummy2"


def test_get_attr():
    """Test `get_attr` and `get_optional_attr` methods."""
    ctx = Context()
    ctx.load_attr_or_type(DummyAttr)

    assert ctx.get_attr("test.dummy_attr") == DummyAttr
    with pytest.raises(UnregisteredConstructException):
        _ = ctx.get_attr("test.dummy_attr2")

    assert ctx.get_optional_attr("test.dummy_attr") == DummyAttr
    assert ctx.get_optional_attr("test.dummy_attr2") is None


def test_get_type():
    """Test `get_type` and `get_optional_type` methods."""
    ctx = Context()
    ctx.load_attr_or_type(DummyAttr)
    ctx.load_attr_or_type(DummyType)

    assert ctx.get_type("test.dummy_type") == DummyType
    with pytest.raises(UnregisteredConstructException):
        _ = ctx.get_type("test.dummy_attr")

    assert ctx.get_optional_type("test.dummy_type") == DummyType
    assert ctx.get_optional_type("test.dummy_attr") is None


def test_get_attr_unregistered():
    """
    Test `get_attr` and `get_optional_attr`
    methods with the `allow_unregistered` flag.
    """
    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(DummyAttr)

    assert ctx.get_optional_attr("test.dummy_attr") == DummyAttr
    attr = ctx.get_optional_attr("test.dummy_attr2")
    assert attr is not None
    assert issubclass(attr, UnregisteredAttr)
    assert not issubclass(attr, TypeAttribute)

    assert ctx.get_attr("test.dummy_attr") == DummyAttr
    assert issubclass(ctx.get_attr("test.dummy_attr2"), UnregisteredAttr)


def test_get_type_unregistered():
    """
    Test `get_type` and `get_optional_type`
    methods with the `allow_unregistered` flag.
    """
    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(DummyType)

    assert ctx.get_optional_type("test.dummy_type") == DummyType
    type = ctx.get_optional_type("test.dummy_type2")
    assert type is not None
    assert issubclass(type, UnregisteredAttr)
    assert issubclass(type, TypeAttribute)

    assert ctx.get_type("test.dummy_type") == DummyType
    assert issubclass(
        ctx.get_type("test.dummy_type2"),
        UnregisteredAttr,
    )


def test_clone_function():
    ctx = Context()
    ctx.load_attr_or_type(DummyAttr)
    ctx.load_op(DummyOp)

    copy = ctx.clone()

    assert ctx == copy
    copy.load_op(DummyOp2)
    assert ctx != copy

    copy = ctx.clone()

    assert ctx == copy
    copy.load_attr_or_type(DummyAttr2)
    assert ctx != copy


def test_register_dialect_get_op_attr():
    ctx = Context()
    ctx.register_dialect("test", lambda: testDialect)
    op = ctx.get_op("test.dummy")
    assert op == DummyOp
    attr = ctx.get_attr("test.dummy_attr")
    assert attr == DummyAttr
    assert "test" in ctx.registered_dialect_names
    assert list(ctx.loaded_dialects) == [testDialect]


def test_register_dialect_already_registered():
    ctx = Context()
    ctx.register_dialect("test", lambda: testDialect)
    with pytest.raises(
        AlreadyRegisteredConstructException,
        match="'test' dialect is already registered",
    ):
        ctx.register_dialect("test", lambda: testDialect2)


def test_register_dialect_already_loaded():
    ctx = Context()
    ctx.load_dialect(testDialect)
    with pytest.raises(
        AlreadyRegisteredConstructException,
        match="'test' dialect is already registered",
    ):
        ctx.register_dialect("test", lambda: testDialect2)


def test_load_registered_dialect():
    ctx = Context()
    ctx.register_dialect("test", lambda: testDialect)
    assert list(ctx.loaded_dialects) == []
    assert list(ctx.registered_dialect_names) == ["test"]
    ctx.load_registered_dialect("test")
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]
    assert list(ctx.loaded_types) == [DummyType]
    assert list(ctx.loaded_attrs) == [DummyAttr]


def test_load_registered_dialect_not_registered():
    ctx = Context()
    with pytest.raises(
        UnregisteredConstructException, match="'test' dialect is not registered"
    ):
        ctx.load_registered_dialect("test")


def test_load_dialect():
    ctx = Context()
    ctx.load_dialect(testDialect)
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]


def test_load_dialect_already_loaded():
    ctx = Context()
    ctx.load_dialect(testDialect)
    with pytest.raises(
        AlreadyRegisteredConstructException,
        match="'test' dialect is already registered",
    ):
        ctx.load_dialect(testDialect)


def test_load_dialect_already_registered():
    ctx = Context()
    ctx.register_dialect("test", lambda: testDialect)
    with pytest.raises(
        AlreadyRegisteredConstructException,
        match="'test' dialect is already registered, use "
        "'load_registered_dialect' instead",
    ):
        ctx.load_dialect(testDialect)


def test_get_dialect():
    ctx = Context()
    assert ctx.get_optional_dialect("test") is None
    ctx.register_dialect("test", lambda: testDialect)
    assert ctx.get_optional_dialect("test") is testDialect


def test_get_optional_op_registered():
    ctx = Context()
    ctx.register_dialect("test", lambda: testDialect)
    assert ctx.get_optional_op("test.dummy") == DummyOp
    assert ctx.get_optional_op("test.dummy2") is None
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]


def test_get_optional_attr_registered():
    ctx = Context()
    ctx.register_dialect("test", lambda: testDialect)
    assert ctx.get_optional_attr("test.dummy_attr") == DummyAttr
    assert ctx.get_optional_attr("test.dummy_attr2") is None
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]


def test_attr_type_same_name():
    """
    Check that a type and an attribute can have the same name.
    """
    ctx = Context()

    class SameNameAttr(ParametrizedAttribute):
        name = "test.same_name"

    class SameNameType(ParametrizedAttribute, TypeAttribute):
        name = "test.same_name"

    dialect = Dialect("test", [], [SameNameAttr, SameNameType])
    ctx.load_dialect(dialect)

    assert ctx.get_attr("test.same_name") == SameNameAttr
    assert ctx.get_type("test.same_name") == SameNameType
