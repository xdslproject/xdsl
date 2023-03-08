import pytest

from xdsl.dialects.builtin import (DenseArrayBase, DenseIntOrFPElementsAttr,
                                   i32, f32, FloatAttr, ArrayAttr, IntAttr,
                                   FloatData, SymbolRefAttr)
from xdsl.utils.exceptions import VerifyException


def test_DenseIntOrFPElementsAttr_fp_type_conversion():
    check1 = DenseIntOrFPElementsAttr.tensor_from_list([4, 5], f32)

    value1 = check1.data.data[0].value.data
    value2 = check1.data.data[1].value.data

    # Ensure type conversion happened properly during attribute construction.
    assert type(value1) == float
    assert value1 == 4.0
    assert type(value2) == float
    assert value2 == 5.0

    t1 = FloatAttr(4.0, f32)
    t2 = FloatAttr(5.0, f32)

    check2 = DenseIntOrFPElementsAttr.tensor_from_list([t1, t2], f32)

    value3 = check2.data.data[0].value.data
    value4 = check2.data.data[1].value.data

    # Ensure type conversion happened properly during attribute construction.
    assert type(value3) == float
    assert value3 == 4.0
    assert type(value4) == float
    assert value4 == 5.0


def test_DenseArrayBase_verifier_failure():
    # Check that a malformed attribute raises a verify error

    with pytest.raises(VerifyException) as err:
        DenseArrayBase([f32, ArrayAttr([IntAttr(0)])])
    assert err.value.args[0] == ("dense array of float element type "
                                 "should only contain floats")

    with pytest.raises(VerifyException) as err:
        DenseArrayBase([i32, ArrayAttr([FloatData(0.0)])])
    assert err.value.args[0] == ("dense array of integer element type "
                                 "should only contain integers")


@pytest.mark.parametrize('ref,expected', (
    (SymbolRefAttr('test'), 'test'),
    (SymbolRefAttr('test', ["2"]), 'test.2'),
    (SymbolRefAttr('test', ["2", "3"]), 'test.2.3'),
))
def test_SymbolRefAttr_string_value(ref: SymbolRefAttr, expected: str):
    assert ref.string_value() == expected


def test_array_len_attr():
    arr = ArrayAttr([IntAttr(i) for i in range(10)])

    assert len(arr) == 10
    assert len(arr.data) == len(arr)
