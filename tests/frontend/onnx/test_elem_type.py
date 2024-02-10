import pytest

from xdsl.frontend.onnx.elem_type import f32, f64, get_elem_type


def test_get_elem_type():
    # test case 1: check if 1 corresponds to f32
    assert get_elem_type(1) == f32

    # test case 11: check if 11 corresponds to f64
    assert get_elem_type(11) == f64

    # test case -1: check if -1 (or other illegal values) corresponds to None
    with pytest.raises(ValueError, match="Unknown elem_type: -1"):
        get_elem_type(-1)
