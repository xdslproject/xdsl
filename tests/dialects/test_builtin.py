import math
import re
from collections.abc import Sequence

import pytest

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AnyTensorType,
    ArrayAttr,
    BFloat16Type,
    BytesAttr,
    ComplexType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    ShapedType,
    Signedness,
    StridedLayoutAttr,
    SymbolRefAttr,
    UnrealizedConversionCastOp,
    VectorBaseTypeAndRankConstraint,
    VectorBaseTypeConstraint,
    VectorRankConstraint,
    VectorType,
    f32,
    f64,
    i1,
    i8,
    i16,
    i32,
    i64,
)
from xdsl.ir import Attribute
from xdsl.irdl import ConstraintContext
from xdsl.utils.exceptions import VerifyException


def test_FloatType_bitwidths():
    assert BFloat16Type().bitwidth == 16
    assert Float16Type().bitwidth == 16
    assert Float32Type().bitwidth == 32
    assert Float64Type().bitwidth == 64
    assert Float80Type().bitwidth == 80
    assert Float128Type().bitwidth == 128


def test_FloatType_formats():
    with pytest.raises(NotImplementedError):
        BFloat16Type().format
    with pytest.raises(NotImplementedError):
        Float16Type().format
    assert Float32Type().format == "<f"
    assert Float64Type().format == "<d"
    with pytest.raises(NotImplementedError):
        Float80Type().format
    with pytest.raises(NotImplementedError):
        Float128Type().format


def test_IntegerType_formats():
    with pytest.raises(NotImplementedError):
        IntegerType(2).format
    assert IntegerType(1).format == "<b"
    assert IntegerType(8).format == "<b"
    assert IntegerType(16).format == "<h"
    assert IntegerType(32).format == "<i"
    assert IntegerType(64).format == "<q"


def test_IndexType_formats():
    assert IndexType().format == "<q"


def test_FloatType_packing():
    nums = (-128, -1, 0, 1, 127)
    buffer = f32.pack(nums)
    unpacked = f32.unpack(buffer, len(nums))
    assert nums == unpacked

    pi = f64.unpack(f64.pack((math.pi,)), 1)[0]
    assert pi == math.pi


def test_IntegerType_size():
    assert IntegerType(1).size == 1
    assert IntegerType(2).size == 1
    assert IntegerType(8).size == 1
    assert IntegerType(16).size == 2
    assert IntegerType(32).size == 4
    assert IntegerType(64).size == 8


def test_IntegerType_normalized():
    si8 = IntegerType(8, Signedness.SIGNED)
    ui8 = IntegerType(8, Signedness.UNSIGNED)

    assert i8.normalized_value(IntAttr(-1)) == IntAttr(-1)
    assert i8.normalized_value(IntAttr(1)) == IntAttr(1)
    assert i8.normalized_value(IntAttr(255)) == IntAttr(-1)

    assert si8.normalized_value(IntAttr(-1)) == IntAttr(-1)
    assert si8.normalized_value(IntAttr(1)) == IntAttr(1)
    assert si8.normalized_value(IntAttr(255)) is None

    assert ui8.normalized_value(IntAttr(-1)) is None
    assert ui8.normalized_value(IntAttr(1)) == IntAttr(1)
    assert ui8.normalized_value(IntAttr(255)) == IntAttr(255)


def test_IntegerAttr_normalize():
    """
    Test that the value within the accepted signless range is normalized to signed
    range.
    """
    assert IntegerAttr(-1, 8) == IntegerAttr(255, 8)
    assert str(IntegerAttr(255, 8)) == "-1 : i8"

    with pytest.raises(
        VerifyException,
        match=re.escape(
            "Integer value -129 is out of range for type i8 which supports "
            "values in the range [-128, 256)"
        ),
    ):
        IntegerAttr(-129, 8)

    with pytest.raises(
        VerifyException,
        match=re.escape(
            "Integer value 256 is out of range for type i8 which supports "
            "values in the range [-128, 256)"
        ),
    ):
        IntegerAttr(256, 8)


def test_IntegerType_packing():
    # i1
    nums_i1 = (0, 1, 0, 1)
    buffer_i1 = i1.pack(nums_i1)
    unpacked_i1 = i1.unpack(buffer_i1, len(nums_i1))
    assert nums_i1 == unpacked_i1

    # i8
    nums_i8 = (-128, -1, 0, 1, 127)
    buffer_i8 = i8.pack(nums_i8)
    unpacked_i8 = i8.unpack(buffer_i8, len(nums_i8))
    assert nums_i8 == unpacked_i8

    # i16
    nums_i16 = (-32768, -1, 0, 1, 32767)
    buffer_i16 = i16.pack(nums_i16)
    unpacked_i16 = i16.unpack(buffer_i16, len(nums_i16))
    assert nums_i16 == unpacked_i16

    # i32
    nums_i32 = (-2147483648, -1, 0, 1, 2147483647)
    buffer_i32 = i32.pack(nums_i32)
    unpacked_i32 = i32.unpack(buffer_i32, len(nums_i32))
    assert nums_i32 == unpacked_i32

    # i64
    nums_i64 = (-9223372036854775808, -1, 0, 1, 9223372036854775807)
    buffer_i64 = i64.pack(nums_i64)
    unpacked_i64 = i64.unpack(buffer_i64, len(nums_i64))
    assert nums_i64 == unpacked_i64

    # f32
    nums_f32 = (-3.140000104904175, -1.0, 0.0, 1.0, 3.140000104904175)
    buffer_f32 = f32.pack(nums_f32)
    unpacked_f32 = f32.unpack(buffer_f32, len(nums_f32))
    assert nums_f32 == unpacked_f32

    # f64
    nums_f64 = (-3.14159265359, -1.0, 0.0, 1.0, 3.14159265359)
    buffer_f64 = f64.pack(nums_f64)
    unpacked_f64 = f64.unpack(buffer_f64, len(nums_f64))
    assert nums_f64 == unpacked_f64

    # Test error cases
    # Different Python versions have different error messages for these
    with pytest.raises(Exception, match="format requires -128 <= number <= 127"):
        # Values must be normalized before packing
        i8.pack((255,))
    with pytest.raises(
        Exception,
        match="format requires (-32768)|(\\(-0x7fff - 1\\)) <= number <= (32767)|(0x7fff)",
    ):
        i16.pack((32768,))
    with pytest.raises(
        Exception, match="format requires -2147483648 <= number <= 2147483647"
    ):
        i32.pack((2147483648,))
    with pytest.raises(
        Exception,
        match="argument out of range|format requires -9223372036854775808 <= number <= 9223372036854775807",
    ):
        i64.pack((9223372036854775808,))


def test_DenseIntOrFPElementsAttr_fp_type_conversion():
    check1 = DenseIntOrFPElementsAttr.tensor_from_list([4, 5], f32, [])

    value1 = check1.get_attrs()[0].value.data
    value2 = check1.get_attrs()[1].value.data

    # Ensure type conversion happened properly during attribute construction.
    assert isinstance(value1, float)
    assert value1 == 4.0
    assert isinstance(value2, float)
    assert value2 == 5.0

    t1 = FloatAttr(4.0, f32)
    t2 = FloatAttr(5.0, f32)

    check2 = DenseIntOrFPElementsAttr.tensor_from_list([t1, t2], f32, [])

    value3 = check2.get_attrs()[0].value.data
    value4 = check2.get_attrs()[1].value.data

    # Ensure type conversion happened properly during attribute construction.
    assert isinstance(value3, float)
    assert value3 == 4.0
    assert isinstance(value4, float)
    assert value4 == 5.0


def test_DenseIntOrFPElementsAttr_from_list():
    attr = DenseIntOrFPElementsAttr.tensor_from_list([5.5], f32, [])

    assert attr.type == AnyTensorType(f32, [])


@pytest.mark.parametrize(
    "ref,expected",
    [
        (SymbolRefAttr("test"), "test"),
        (SymbolRefAttr("test", ["2"]), "test.2"),
        (SymbolRefAttr("test", ["2", "3"]), "test.2.3"),
    ],
)
def test_SymbolRefAttr_string_value(ref: SymbolRefAttr, expected: str):
    assert ref.string_value() == expected


def test_array_len_and_iter_attr():
    arr = ArrayAttr([IntAttr(i) for i in range(10)])

    assert len(arr) == 10
    assert len(arr.data) == len(arr)

    # check that it is iterable
    assert tuple(arr) == arr.data


@pytest.mark.parametrize(
    "attr, dims, num_scalable_dims",
    [
        (i32, (1, 2), 0),
        (i32, (1, 2), 1),
        (i32, (1, 1, 3), 0),
        (i64, (1, 1, 3), 2),
        (i64, (), 0),
    ],
)
def test_vector_constructor(attr: Attribute, dims: list[int], num_scalable_dims: int):
    vec = VectorType(attr, dims, num_scalable_dims)

    assert vec.get_num_dims() == len(dims)
    assert vec.get_num_scalable_dims() == num_scalable_dims
    assert vec.get_shape() == dims


@pytest.mark.parametrize(
    "dims, num_scalable_dims",
    [
        ([], 1),
        ([1, 2], 3),
        ([1], 2),
    ],
)
def test_vector_verifier_fail(dims: list[int], num_scalable_dims: int):
    with pytest.raises(VerifyException):
        VectorType(i32, dims, num_scalable_dims)

    with pytest.raises(VerifyException):
        VectorType(i32, dims, -1)


def test_vector_rank_constraint_verify():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorRankConstraint(2)

    constraint.verify(vector_type, ConstraintContext())


def test_vector_rank_constraint_rank_mismatch():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorRankConstraint(3)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type, ConstraintContext())
    assert e.value.args[0] == "Expected vector rank to be 3, got 2."


def test_vector_rank_constraint_attr_mismatch():
    memref_type = MemRefType(i32, [1, 2])
    constraint = VectorRankConstraint(3)

    with pytest.raises(VerifyException) as e:
        constraint.verify(memref_type, ConstraintContext())
    assert e.value.args[0] == "memref<1x2xi32> should be of type VectorType."


def test_vector_base_type_constraint_verify():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorBaseTypeConstraint(i32)

    constraint.verify(vector_type, ConstraintContext())


def test_vector_base_type_constraint_type_mismatch():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorBaseTypeConstraint(i64)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type, ConstraintContext())
    assert e.value.args[0] == "Expected vector type to be i64, got i32."


def test_vector_base_type_constraint_attr_mismatch():
    memref_type = MemRefType(i32, [1, 2])
    constraint = VectorBaseTypeConstraint(i32)

    with pytest.raises(VerifyException) as e:
        constraint.verify(memref_type, ConstraintContext())
    assert e.value.args[0] == "memref<1x2xi32> should be of type VectorType."


def test_vector_base_type_and_rank_constraint_verify():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i32, 2)

    constraint.verify(vector_type, ConstraintContext())


def test_vector_base_type_and_rank_constraint_base_type_mismatch():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i64, 2)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type, ConstraintContext())
    assert e.value.args[0] == "Expected vector type to be i64, got i32."


def test_vector_base_type_and_rank_constraint_rank_mismatch():
    vector_type = VectorType(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i32, 3)

    with pytest.raises(VerifyException) as e:
        constraint.verify(vector_type, ConstraintContext())
    assert e.value.args[0] == "Expected vector rank to be 3, got 2."


def test_vector_base_type_and_rank_constraint_attr_mismatch():
    memref_type = MemRefType(i32, [1, 2])
    constraint = VectorBaseTypeAndRankConstraint(i32, 2)

    error_msg = """The following constraints were not satisfied:
memref<1x2xi32> should be of type VectorType.
memref<1x2xi32> should be of type VectorType."""

    with pytest.raises(VerifyException) as e:
        constraint.verify(memref_type, ConstraintContext())
    assert e.value.args[0] == error_msg


def test_unrealized_conversion_cast():
    i64_constant = ConstantOp.from_int_and_width(1, 64)
    f32_constant = ConstantOp(FloatAttr(10.1, f32))

    conv_op1 = UnrealizedConversionCastOp.get([i64_constant.results[0]], [f32])
    conv_op2 = UnrealizedConversionCastOp.get([f32_constant.results[0]], [i32])

    assert conv_op1.inputs[0].type == i64
    assert conv_op1.outputs[0].type == f32

    assert conv_op2.inputs[0].type == f32
    assert conv_op2.outputs[0].type == i32


@pytest.mark.parametrize(
    "strides, offset, expected_strides, expected_offset",
    [
        ([2], None, ArrayAttr([IntAttr(2)]), NoneAttr()),
        ([None], 2, ArrayAttr([NoneAttr()]), IntAttr(2)),
        ([IntAttr(2)], NoneAttr(), ArrayAttr([IntAttr(2)]), NoneAttr()),
        ([NoneAttr()], IntAttr(2), ArrayAttr([NoneAttr()]), IntAttr(2)),
    ],
)
def test_strided_constructor(
    strides: ArrayAttr[IntAttr | NoneAttr] | Sequence[int | None | IntAttr | NoneAttr],
    offset: int | None | IntAttr | NoneAttr,
    expected_strides: ArrayAttr[IntAttr | NoneAttr],
    expected_offset: IntAttr | NoneAttr,
):
    strided = StridedLayoutAttr(strides, offset)
    assert strided.strides == expected_strides
    assert strided.offset == expected_offset


def test_complex_init():
    assert ComplexType(f32) == ComplexType.new([f32])
    assert ComplexType(i32) == ComplexType.new([i32])


def test_dense_as_tuple():
    floats = DenseArrayBase.from_list(f32, [3.14159, 2.71828])
    assert floats.get_values() == (3.141590118408203, 2.718280076980591)

    ints = DenseArrayBase.from_list(i32, [1, 1, 2, 3, 5, 8])
    assert ints.get_values() == (1, 1, 2, 3, 5, 8)


def test_create_dense_int():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Integer value 99999999 is out of range for type i8 which supports values in the range [-128, 256)"
        ),
    ):
        DenseArrayBase.create_dense_int(i8, (99999999, 255, 256))


def test_create_dense_wrong_size():
    with pytest.raises(
        VerifyException,
        match=re.escape("Data length of array (1) not divisible by element size 2"),
    ):
        DenseArrayBase((i16, BytesAttr(b"F")))


def test_strides():
    assert ShapedType.strides_for_shape(()) == ()
    assert ShapedType.strides_for_shape((), factor=2) == ()
    assert ShapedType.strides_for_shape((1,)) == (1,)
    assert ShapedType.strides_for_shape((1,), factor=2) == (2,)
    assert ShapedType.strides_for_shape((2, 3)) == (3, 1)
    assert ShapedType.strides_for_shape((4, 5, 6), factor=2) == (60, 12, 2)
