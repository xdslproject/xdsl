import pytest

from xdsl.dialects.arm_neon import (
    NeonArrangement,
    NeonArrangementAttr,
    NEONRegisterType,
    VectorWithArrangement,
)
from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type, VectorType


def test_assembly_str_without_index():
    reg = NEONRegisterType.from_name("v0")
    arrangement = NeonArrangementAttr(NeonArrangement.D)
    vecreg = VectorWithArrangement(reg=reg, arrangement=arrangement, index=None)
    assert vecreg.assembly_str() == "v0.2D"


def test_assembly_str_with_index():
    reg = NEONRegisterType.from_name("v0")
    arrangement = NeonArrangementAttr(NeonArrangement.D)
    vecreg = VectorWithArrangement(reg=reg, arrangement=arrangement, index=5)
    assert vecreg.assembly_str() == "v0.D[5]"


def test_arr_from_vec_f16():
    arrangement = NeonArrangement.H
    vectype = VectorType(Float16Type(), [8])
    assert NeonArrangement.from_vec_type(vectype) == arrangement


def test_arr_from_vec_f32():
    arrangement = NeonArrangement.S
    vectype = VectorType(Float32Type(), [4])
    assert NeonArrangement.from_vec_type(vectype) == arrangement


def test_arr_from_vec_f64():
    arrangement = NeonArrangement.D
    vectype = VectorType(Float64Type(), [2])
    assert NeonArrangement.from_vec_type(vectype) == arrangement


def test_arr_from_vec_invalid():
    unsupported_vec = VectorType(Float16Type(), [3])

    with pytest.raises(
        ValueError, match=f"Invalid vector type for ARM NEON: {str(unsupported_vec)}"
    ):
        NeonArrangement.from_vec_type(unsupported_vec)
