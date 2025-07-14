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


def test_arr_from_vec():
    assert (
        NeonArrangement.from_vec_type(VectorType(Float16Type(), [8]))
        == NeonArrangement.H
    )
    assert (
        NeonArrangement.from_vec_type(VectorType(Float32Type(), [4]))
        == NeonArrangement.S
    )
    assert (
        NeonArrangement.from_vec_type(VectorType(Float64Type(), [2]))
        == NeonArrangement.D
    )

    with pytest.raises(
        ValueError, match="Invalid vector type for ARM NEON: vector<3xf16>"
    ):
        NeonArrangement.from_vec_type(VectorType(Float16Type(), [3]))
