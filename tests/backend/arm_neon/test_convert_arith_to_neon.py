import pytest

from xdsl.backend.arm_neon.lowering.convert_arith_to_neon import (
    get_arrangement_from_vec_type,
)
from xdsl.dialects.arm_neon import NeonArrangement
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    VectorType,
)


def test_get_arrangement_from_vec_type_f16():
    vec_type = VectorType(Float16Type(), [8])
    arrangement = NeonArrangement.H
    assert get_arrangement_from_vec_type(vec_type) == arrangement


def test_get_arrangement_from_vec_type_f32():
    vec_type = VectorType(Float32Type(), [4])
    arrangement = NeonArrangement.S
    assert get_arrangement_from_vec_type(vec_type) == arrangement


def test_get_arrangement_from_vec_type_f64():
    vec_type = VectorType(Float64Type(), [2])
    arrangement = NeonArrangement.D
    assert get_arrangement_from_vec_type(vec_type) == arrangement


def test_get_arrangement_from_vec_type_num_els_invalid():
    vec_type = VectorType(Float16Type(), [6])
    with pytest.raises(
        ValueError,
        match="Invalid number of F16 elements in vector: Expected 8, received 6",
    ):
        get_arrangement_from_vec_type(vec_type)
