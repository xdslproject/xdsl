"""
Toy language dialect from MLIR tutorial.
"""

from __future__ import annotations

from typing import Annotated, List, TypeAlias, Union, Optional, Any, cast

from xdsl.ir import Dialect, SSAValue
from xdsl.dialects.builtin import (Float64Type, FunctionType, Attribute,
                                   FlatSymbolRefAttr, TensorType,
                                   UnrankedTensorType, f64,
                                   DenseIntOrFPElementsAttr, AnyTensorType,
                                   StringAttr)
from xdsl.irdl import (OpAttr, Operand, OptOpAttr, OptOperand, VarOpResult,
                       VarOperand, irdl_op_definition, AnyAttr, Block, Region,
                       Operation, OpResult)
from xdsl.utils.exceptions import VerifyException

TensorTypeF64: TypeAlias = TensorType[Float64Type]
UnrankedTensorTypeF64: TypeAlias = UnrankedTensorType[Float64Type]
AnyTensorTypeF64: TypeAlias = TensorTypeF64 | UnrankedTensorTypeF64


@irdl_op_definition
class ConstantOp(Operation):
    """
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
    """
    name: str = "toy.constant"
    value: OpAttr[DenseIntOrFPElementsAttr]
    res: Annotated[OpResult, TensorTypeF64]

    @staticmethod
    def from_list(data: List[float], shape: List[int]):
        value = DenseIntOrFPElementsAttr.tensor_from_list(data, f64, shape)

        return ConstantOp.create(result_types=[value.type],
                                 attributes={"value": value})

    def verify_(self) -> None:
        if not self.res.typ == self.value.type:
            raise VerifyException(
                "Expected value and result types to be equal: "
                f"{self.res.typ}, {self.value.type}")


@irdl_op_definition
class AddOp(Operation):
    """
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
    """
    name: str = 'toy.add'
    arguments: Annotated[VarOperand, AnyTensorTypeF64]
    res: Annotated[OpResult, AnyTensorTypeF64]

    @classmethod
    def from_summands(cls: type[AddOp], lhs: SSAValue, rhs: SSAValue) -> AddOp:
        assert isinstance(lhs.typ, TensorType | UnrankedTensorType)
        element_type = cast(Float64Type,
                            cast(TensorType[Any], lhs.typ).element_type)
        return cls.create(result_types=[element_type], operands=[lhs, rhs])

    def verify_(self):
        if not len(self.arguments):
            raise VerifyException("Expected AddOp args to not be empty")

        shape = None
        for arg in self.arguments:
            # Expect shapes to be the same whenever they are defined, no check for unranked
            if isinstance(arg.typ, TensorType):
                if shape is None:
                    shape = arg.typ.shape
                else:
                    if shape != arg.typ.shape:
                        raise VerifyException(
                            "Expected AddOp args to have the same shape")


@irdl_op_definition
class FuncOp(Operation):
    """
    The "toy.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f64>
      %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      toy.print %1 : tensor<2x2xf64>
      toy.return
    }
    ```
    """
    name = 'toy.func'
    body: Region
    sym_name: OpAttr[StringAttr]
    function_type: OpAttr[FunctionType]
    sym_visibility: OptOpAttr[StringAttr]

    @staticmethod
    def from_region(name: str, ftype: FunctionType, region: Region):
        return FuncOp.create(attributes={
            "sym_name": StringAttr.from_str(name),
            "function_type": ftype,
            "sym_visibility": StringAttr.from_str("private"),
        },
                             regions=[region])

    @staticmethod
    def from_callable(name: str,
                      input_types: List[Attribute],
                      return_types: List[Attribute],
                      func: Block.BlockCallback,
                      private: bool = False):
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": name,
            "function_type": type_attr,
        }
        if private:
            attributes["sym_visibility"] = "private"
        return FuncOp.build(attributes=attributes,
                            regions=[
                                Region.from_block_list(
                                    [Block.from_callable(input_types, func)])
                            ])


@irdl_op_definition
class GenericCallOp(Operation):
    name: str = "toy.generic_call"
    arguments: Annotated[VarOperand, AnyAttr()]
    callee: OpAttr[FlatSymbolRefAttr]

    # Note: naming this results triggers an ArgumentError
    res: Annotated[VarOpResult, AnyTensorTypeF64]

    @classmethod
    def get(cls: type[GenericCallOp], callee: Union[str, FlatSymbolRefAttr],
            operands: List[Union[SSAValue, OpResult]],
            return_types: List[Attribute]) -> GenericCallOp:
        if isinstance(callee, str):
            callee = FlatSymbolRefAttr.from_str(callee)

        return cls.create(operands=operands,
                          result_types=return_types,
                          attributes={"callee": callee})


@irdl_op_definition
class MulOp(Operation):
    """
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
    """
    name: str = 'toy.mul'
    arguments: Annotated[VarOperand, AnyTensorTypeF64]
    res: Annotated[OpResult, AnyTensorTypeF64]

    @classmethod
    def from_summands(cls: type[MulOp], lhs: SSAValue, rhs: SSAValue) -> MulOp:
        return cls.create(result_types=[lhs.typ], operands=[lhs, rhs])

    def verify_(self):
        if not len(self.arguments):
            raise VerifyException("Expected MulOp args to not be empty")

        shape = None
        for arg in self.arguments:
            # Expect shapes to be the same whenever they are defined, no check for unranked
            if isinstance(arg.typ, TensorType):
                if shape is None:
                    shape = arg.typ.shape
                else:
                    if shape != arg.typ.shape:
                        raise VerifyException(
                            "Expected MulOp args to have the same shape")


@irdl_op_definition
class PrintOp(Operation):
    """
    The "print" builtin operation prints a given input tensor, and produces
    no results.
    """
    name: str = 'toy.print'
    arguments: Annotated[VarOperand, AnyAttr()]

    @classmethod
    def from_input(cls: type[PrintOp], input: SSAValue) -> PrintOp:
        return cls.create(operands=[input])


@irdl_op_definition
class ReturnOp(Operation):
    """
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
    """
    name: str = 'toy.return'
    arguments: Annotated[OptOperand, AnyTensorTypeF64]

    @classmethod
    def from_input(cls: type[ReturnOp],
                   input: Optional[SSAValue] = None) -> ReturnOp:
        return cls.create(operands=[input] if input is not None else [])


@irdl_op_definition
class ReshapeOp(Operation):
    """
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
    """
    name: str = 'toy.reshape'
    arguments: Annotated[VarOperand, AnyTensorTypeF64]
    # We expect that the reshape operation returns a statically shaped tensor.
    res: Annotated[OpResult, TensorTypeF64]

    @classmethod
    def from_input(cls: type[ReshapeOp], input: SSAValue,
                   shape: List[int]) -> ReshapeOp:
        assert isinstance(input.typ, TensorType | UnrankedTensorType)
        element_type = cast(Float64Type,
                            cast(TensorType[Any], input.typ).element_type)
        t = AnyTensorType.from_type_and_list(element_type, shape)
        return cls.create(result_types=[t], operands=[input])

    def verify_(self):
        result_type = self.res.typ
        assert isinstance(result_type, TensorType)
        result_type = cast(TensorTypeF64, result_type)
        if not len(result_type.shape.data):
            raise VerifyException(
                'Reshape operation result shape should be defined')


@irdl_op_definition
class TransposeOp(Operation):
    name: str = 'toy.transpose'
    arguments: Annotated[Operand, AnyTensorTypeF64]
    res: Annotated[OpResult, AnyTensorTypeF64]

    @staticmethod
    def from_input(input: SSAValue):
        input_type = input.typ
        assert isinstance(input_type, TensorType | UnrankedTensorType)
        output_type: TensorType[Any] | UnrankedTensorType[Any]
        if isinstance(input_type, TensorType):
            element_type = cast(Float64Type,
                                cast(TensorType[Any], input_type).element_type)
            output_type = TensorType.from_type_and_list(
                element_type, list(reversed(input_type.shape.data)))
        elif isinstance(input_type, UnrankedTensorType):
            output_type = input_type
        else:
            assert False, f'{input_type}: {type(input_type)}'

        return TransposeOp.create(operands=[input], result_types=[output_type])


Toy = Dialect([
    ConstantOp, AddOp, FuncOp, GenericCallOp, PrintOp, MulOp, ReturnOp,
    ReshapeOp, TransposeOp
], [])
