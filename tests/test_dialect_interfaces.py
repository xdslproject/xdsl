import pytest

from xdsl.dialect_interfaces import OpAsmDialectInterface


def test_op_asm_interface():
    interf = OpAsmDialectInterface()
    key = interf.parse_resource("some_key", "0x0800000001")
    assert key == "some_key"
    assert key in interf.blob_storage

    key = interf.parse_resource("some_key", "0x0800000002")
    assert key == "some_key_0"
    assert key in interf.blob_storage

    with pytest.raises(ValueError):
        interf.parse_resource("other_key", "normal string")
