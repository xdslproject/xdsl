import pytest

from xdsl.dialect_interfaces import OpAsmDialectInterface


def test_op_asm_interface():
    interf = OpAsmDialectInterface()

    key = interf.declare_resource("some_key")
    assert key == "some_key"
    interf.parse_resource(key, "0x0800000001")
    assert interf.blob_storage[key] == "0x0800000001"

    key = interf.declare_resource("some_key")
    assert key == "some_key_0"
    interf.parse_resource(key, "0x0800000002")
    assert interf.blob_storage[key] == "0x0800000002"

    with pytest.raises(ValueError):
        interf.parse_resource("some_key", "normal string")

    with pytest.raises(KeyError):
        interf.parse_resource("non existent key", "0x0800000003")
