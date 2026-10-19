import pytest

from xdsl.dialect_interfaces.constant_materialization import (
    ConstantMaterializationInterface,
)
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    TensorType,
    f32,
    i32,
)
from xdsl.dialects.csl import (
    CSL,
    Add16Op,
    ComptimeStringType,
    ConstStrOp,
    DsdKind,
    DsdType,
    GetMemDsdOp,
)
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

tensor = create_ssa_value(TensorType(f32, [4]))
size_i32 = create_ssa_value(IntegerType(32, Signedness.SIGNED))
dest_dsd = GetMemDsdOp(
    operands=[tensor, size_i32], result_types=[DsdType(DsdKind.mem1d_dsd)]
)
src_dsd1 = GetMemDsdOp(
    operands=[tensor, size_i32], result_types=[DsdType(DsdKind.mem1d_dsd)]
)
src_dsd2 = GetMemDsdOp(
    operands=[tensor, size_i32], result_types=[DsdType(DsdKind.mem1d_dsd)]
)
i16_value = create_ssa_value(IntegerType(16, Signedness.SIGNED))
u16_value = create_ssa_value(IntegerType(16, Signedness.UNSIGNED))


def test_verify_valid_builtin_signature():
    Add16Op(operands=[(dest_dsd, src_dsd1, src_dsd2)], result_types=[]).verify_()
    Add16Op(operands=[(dest_dsd, i16_value, src_dsd1)], result_types=[]).verify_()
    Add16Op(operands=[(dest_dsd, u16_value, src_dsd1)], result_types=[]).verify_()
    Add16Op(operands=[(dest_dsd, src_dsd1, i16_value)], result_types=[]).verify_()
    Add16Op(operands=[(dest_dsd, src_dsd1, u16_value)], result_types=[]).verify_()


def test_verify_invalid_builtin_signature():
    with pytest.raises(VerifyException):
        Add16Op(
            operands=[(dest_dsd, src_dsd1, src_dsd2, dest_dsd)], result_types=[]
        ).verify_()
    with pytest.raises(VerifyException):
        Add16Op(operands=[(dest_dsd, src_dsd1)], result_types=[]).verify_()
    with pytest.raises(VerifyException):
        Add16Op(operands=[(dest_dsd, i16_value, u16_value)], result_types=[]).verify_()
    with pytest.raises(VerifyException):
        Add16Op(operands=[(i16_value, src_dsd1, u16_value)], result_types=[]).verify_()
    with pytest.raises(VerifyException):
        Add16Op(operands=[(dest_dsd, src_dsd1, size_i32)], result_types=[]).verify_()


def test_const_str_fold():
    op = ConstStrOp("hello")
    assert op.fold() == (StringAttr("hello"),)
    assert ConstantLike.get_constant_value(op.res) == StringAttr("hello")


def test_constant_materialization():
    interface = CSL.get_interface(ConstantMaterializationInterface)
    assert interface is not None
    const = interface.materialize_constant(StringAttr("hello"), ComptimeStringType())
    assert isinstance(const, ConstStrOp)
    assert const.value == StringAttr("hello")
    assert const.result_types[0] == ComptimeStringType()

    with pytest.raises(ValueError, match="Cannot materialize constant"):
        interface.materialize_constant(IntegerAttr(42, 32), i32)
    with pytest.raises(ValueError, match="Cannot materialize constant"):
        interface.materialize_constant(StringAttr("hello"), i32)
