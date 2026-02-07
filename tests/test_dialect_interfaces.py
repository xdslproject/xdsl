import pytest

from xdsl.dialect_interfaces import DialectInterface
from xdsl.dialect_interfaces.op_asm import OpAsmDialectInterface
from xdsl.ir import Dialect


class TestInterface(DialectInterface):
    pass


def test_interfaces():
    dialect = Dialect("dialect", [], [], [TestInterface()])
    another_dialect = Dialect("another", [], [], [])

    assert dialect.has_interface(TestInterface)
    assert not another_dialect.has_interface(TestInterface)
    assert isinstance(dialect.get_interface(TestInterface), TestInterface)


def test_op_asm_interface():
    interf = OpAsmDialectInterface()

    key = interf.declare_resource("some_key")
    assert key == "some_key"
    interf.parse_resource(key, "0x0800000001")
    assert interf.lookup(key) == "0x0800000001"

    key = interf.declare_resource("some_key")
    assert key == "some_key_0"
    interf.parse_resource(key, "0x0800000002")
    assert interf.lookup(key) == "0x0800000002"

    with pytest.raises(
        ValueError, match="Blob must be a hex string, got: normal string"
    ):
        interf.parse_resource("some_key", "normal string")

    with pytest.raises(KeyError):
        interf.parse_resource("non existent key", "0x0800000003")

    interf.declare_resource("non_assigned_key")
    assert interf.build_resources(["some_key", "non_assigned_key"]) == {
        "some_key": "0x0800000001"
    }
