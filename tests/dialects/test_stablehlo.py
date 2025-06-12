from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    ComplexType,
    DenseIntOrFPElementsAttr,
    IntAttr,
    IntegerType,
    RankedStructure,
    TensorType,
    i1,
    i32,
    f32,
    i64,
)
from xdsl.dialects.stablehlo import ConstantOp


def test_constant_construction():
    c1 = ConstantOp([1])
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == i32
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])


def test_constant_construction_different_type():
    c1 = ConstantOp([1], element_ty=i64)
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == i64
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])


def test_constant_construction_different_shape():
    c1 = ConstantOp([1, 2, 3, 4], shape=[2, 2])
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == i32
    assert tensor_ty.shape == ArrayAttr([IntAttr(2), IntAttr(2)])


def test_constant_construction_infer_from_bool():
    c1 = ConstantOp([True])
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == i1
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])


def test_constant_construction_infer_from_float():
    c1 = ConstantOp([1.0])
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == f32
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])


def test_constant_construction_infer_from_tuple_int():
    c1 = ConstantOp([(1, 0)])
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == ComplexType(i32)
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])


def test_constant_construction_infer_from_tuple_float():
    c1 = ConstantOp([(1.0, 0.0)])
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == ComplexType(f32)
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])


def test_constant_construction_from_dense_int_or_fp_elements_attr():
    type = TensorType(i32, [1])
    type = cast(RankedStructure[IntegerType], type)
    c1attr = DenseIntOrFPElementsAttr.create_dense_int(type, [1])
    c1 = ConstantOp(c1attr)
    assert isinstance(tensor_ty := c1.value.type, TensorType)
    assert tensor_ty.element_type == i32
    assert tensor_ty.shape == ArrayAttr([IntAttr(1)])
