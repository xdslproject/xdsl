from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, Container, List
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntegerAttr, TensorType
from xdsl.frontend_deprecated.dialects.builtin import IntegerType
from xdsl.ir import Attribute, Dialect, OpResult, Operation, Region, SSAValue
from xdsl.irdl import AnyAttr, AnyOf, AttributeDef, Operand, RegionDef, VarOperand, VarOperandDef, irdl_op_definition
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class Cast(Operation):
    name: str = "tensor.cast"
    value: Annotated[Operand, TensorType]
    result: Annotated[OpResult, TensorType]

    def verify_(self) -> None:
        if self.value.typ.element_type != self.result.typ.element_type:
            raise VerifyException("Result and input types must match")
        if self.value.typ.get_num_dims() != self.result.typ.get_num_dims():
            raise VerifyException("Result and input tensors must have the same number of dimensions")

        num_dims = self.value.typ.get_num_dims()
        input_shape = self.value.typ.get_shape()
        output_shape = self.value.typ.get_shape()
        for i in range(num_dims):
            if input_shape[i] != -1 and output_shape[i] != -1 and input_shape[i] != output_shape[i]:
                raise VerifyException("Result and input tensors must have the same static dimensions")

    @staticmethod
    def get(value: SSAValue | Operation, dst_type: TensorType) -> Cast:
        return Cast.build(operands=[SSAValue.get(value)], result_types=[dst_type])


@irdl_op_definition
class Empty(Operation):
    # TODO: fix naming and support dynamic tensors.
    name: str = "linalg.init_tensor"
    result: Annotated[OpResult, TensorType]
    static_sizes = AttributeDef(ArrayAttr)

    @staticmethod
    def get(static_sizes: List[int], el_ty: Attribute) -> Empty:
        return Empty.create(
            result_types=[TensorType.from_type_and_list(el_ty, static_sizes)],
            attributes={
                "static_sizes": ArrayAttr.from_list([IntegerAttr.from_int_and_width(s, 64) for s in static_sizes])
            })


@irdl_op_definition
class Generate(Operation):
    name: str = "tensor.generate"
    arguments: Annotated[VarOperand, Attribute]
    res: Annotated[OpResult, TensorType]

    body = RegionDef()

    # def verify_(self) -> None:
    #     pass

    @staticmethod
    def from_region(bounds: list[Operation | SSAValue], region: Region, dims: List[int], el_ty: Attribute) -> Generate:
        return Generate.build(operands=[bounds],
                         result_types=[TensorType.from_type_and_list(el_ty, dims)],
                         regions=[region])


@irdl_op_definition
class Splat(Operation):
    name: str = "tensor.splat"
    value: Annotated[Operand, Attribute]
    result: Annotated[OpResult, TensorType]

    @staticmethod
    def get(op: Operation | SSAValue, dims: List[int]) -> Splat:
        return Splat.build(operands=[op], result_types=[TensorType.from_type_and_list(op.typ, dims)])


@irdl_op_definition
class Extract(Operation):
    name = "tensor.extract"
    tensor: Annotated[Operand, TensorType]
    indices: Annotated[VarOperand, IndexType]
    res: Annotated[OpResult, AnyAttr()]

    # def verify_(self):
    #     if self.tensor.typ.element_type != self.res.typ:
    #         raise Exception(
    #             "expected return type to match the tensor element type")

    #     if self.tensor.typ.get_num_dims() != len(self.indices):
    #         raise Exception("expected an index for each dimension")

    @staticmethod
    def get(tensor: SSAValue | Operation,
            *indices: SSAValue | Operation) -> Extract:
        operands = [tensor] + [SSAValue.get(i) for i in indices]
        return Extract.create(operands, result_types=[SSAValue.get(tensor).typ.element_type])


@irdl_op_definition
class Insert(Operation):
    name = "tensor.insert"
    value: Annotated[Operand, Attribute]
    tensor: Annotated[Operand, TensorType]
    indices: Annotated[VarOperand, IndexType]

    res: Annotated[OpResult, TensorType]

    # def verify_(self):
    #     if self.tensor.typ.element_type != self.value.typ:
    #         raise Exception(
    #             "expected value type to match the tensor element type")

    #     if self.tensor.typ.get_num_dims() != len(self.indices):
    #         raise Exception("expected an index for each dimension")

    @staticmethod
    def get(value: Operation | SSAValue, tensor: Operation | SSAValue,
            *indices: Operation | SSAValue) -> 'Insert':
        operands = [value, tensor] + [SSAValue.get(i) for i in indices]
        return Insert.create(operands, result_types=[tensor.typ])


@irdl_op_definition
class Yield(Operation):
    name: str = "tensor.yield"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(
            operands=[SSAValue.get(operand) for operand in operands])

Tensor = Dialect([Cast, Empty, Generate, Extract, Insert, Splat, Yield], [])
