import pytest

from xdsl.dialects.builtin import (DenseArrayBase, DenseIntOrFPElementsAttr,
                                   i32, f32, FloatAttr, ArrayAttr, IntAttr,
                                   FloatData, SymbolRefAttr,
                                   VectorBaseTypeConstraint,
                                   VectorRankConstraint,
                                   VectorBaseTypeAndRankConstraint)
from xdsl.dialects.builtin import i32, i64, VectorType, UnrealizedConversionCastOp
from xdsl.dialects.arith import Constant
from xdsl.dialects.memref import MemRefType
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


def test_vector_rank_constraint_verify():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorRankConstraint(2)

    constraint.verify(vector_type)


def test_vector_rank_constraint_rank_mismatch():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorRankConstraint(3)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type)
    assert e.value.args[0] == "Expected vector rank to be 3, got 2."


def test_vector_rank_constraint_attr_mismatch():
    memref_type = MemRefType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorRankConstraint(3)

    with pytest.raises(VerifyException) as e:
        constraint.verify(memref_type)
    assert e.value.args[
        0] == "!memref<[1 : !index, 2 : !index], !i32> should be of type VectorType."


def test_vector_base_type_constraint_verify():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeConstraint(i32)

    constraint.verify(vector_type)


def test_vector_base_type_constraint_type_mismatch():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeConstraint(i64)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type)
    assert e.value.args[0] == "Expected vector type to be !i64, got !i32."


def test_vector_base_type_constraint_attr_mismatch():
    memref_type = MemRefType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeConstraint(i32)

    with pytest.raises(VerifyException) as e:
        constraint.verify(memref_type)
    assert e.value.args[
        0] == "!memref<[1 : !index, 2 : !index], !i32> should be of type VectorType."


def test_vector_base_type_and_rank_constraint_verify():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i32, 2)

    constraint.verify(vector_type)


def test_vector_base_type_and_rank_constraint_base_type_mismatch():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i64, 2)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type)
    assert e.value.args[0] == "Expected vector type to be !i64, got !i32."


def test_vector_base_type_and_rank_constraint_rank_mismatch():
    vector_type = VectorType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i32, 3)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type)
    assert e.value.args[0] == "Expected vector rank to be 3, got 2."


def test_vector_base_type_and_rank_constraint_attr_mismatch():
    memref_type = MemRefType.from_element_type_and_shape(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i32, 2)

    error_msg = """The following constraints were not satisfied:
!memref<[1 : !index, 2 : !index], !i32> should be of type VectorType.
!memref<[1 : !index, 2 : !index], !i32> should be of type VectorType."""

    with pytest.raises(VerifyException) as e:
        constraint.verify(memref_type)
    assert e.value.args[0] == error_msg


def test_unrealized_conversion_cast():
    i64_constant = Constant.from_int_and_width(1, 64)
    f32_constant = Constant.from_float_and_width(10.1, f32)

    conv_op1 = UnrealizedConversionCastOp.get(i64_constant.results[0], f32)
    conv_op2 = UnrealizedConversionCastOp.get(f32_constant.results[0], i32)

    assert (conv_op1.input.typ == i64)
    assert (conv_op1.output.typ == f32)

    assert (conv_op2.input.typ == f32)
    assert (conv_op2.output.typ == i32)
