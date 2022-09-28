'''
Toy language dialect from MLIR tutorial.
'''

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-builtin

from dataclasses import dataclass
from typing import List, Union, TypeVar, Optional, Any
from xmlrpc.client import Boolean

from xdsl.ir import MLContext, SSAValue
from xdsl.dialects.builtin import (FunctionType, Attribute, FlatSymbolRefAttr,
                                   TensorType, UnrankedTensorType, f64,
                                   DenseIntOrFPElementsAttr, SymbolNameAttr,
                                   AnyTensorType, StringAttr)
from xdsl.irdl import (irdl_op_definition, VarOperandDef, AnyAttr, Block,
                       RegionDef, Region, Operation, AttributeDef,
                       VarResultDef, ResultDef, OpResult, OptOperandDef,
                       OperandDef, OptAttributeDef)


@dataclass
class Toy:
    '''
    Toy language dialect from MLIR tutorial.
    '''
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(ConstantOp)
        self.ctx.register_op(AddOp)
        self.ctx.register_op(FuncOp)
        self.ctx.register_op(GenericCallOp)
        self.ctx.register_op(PrintOp)
        self.ctx.register_op(MulOp)
        self.ctx.register_op(ReturnOp)
        self.ctx.register_op(ReshapeOp)
        self.ctx.register_op(TransposeOp)


TensorTypeF64 = TensorType.from_type_and_list(f64)


@irdl_op_definition
class ConstantOp(Operation):
    '''
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
    '''
    name: str = "toy.constant"
    result = ResultDef(AnyAttr())  #TensorType[F64])
    value = AttributeDef(AnyAttr())  #F64ElementsAttr

    # TODO verify that the result and value type are equal

    @staticmethod
    def from_list(data: List[float], shape: List[int]):
        value = DenseIntOrFPElementsAttr.tensor_from_list(data, f64, shape)
        return ConstantOp.create(result_types=[value.type],
                                 attributes={"value": value})


AddOpT = TypeVar('AddOpT', bound='AddOp')


@irdl_op_definition
class AddOp(Operation):
    '''
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
    '''
    name: str = 'toy.add'
    arguments = VarOperandDef(AnyAttr())
    result = ResultDef(AnyAttr())  # F64ElementsAttr

    @classmethod
    def from_summands(cls: type[AddOpT], lhs: OpResult,
                      rhs: SSAValue) -> AddOpT:
        return super().create(result_types=[lhs.typ.element_type],
                              operands=[lhs, rhs])


@irdl_op_definition
class FuncOp(Operation):
    '''
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
    '''
    name = 'toy.func'
    body = RegionDef()
    sym_name = AttributeDef(SymbolNameAttr)
    function_type = AttributeDef(FunctionType)
    sym_visibility = OptAttributeDef(StringAttr)

    @staticmethod
    def from_region(name: str, ftype: FunctionType, region: Region):
        return FuncOp.create(result_types=[ftype.outputs],
                             attributes={
                                 "sym_name": SymbolNameAttr.from_str(name),
                                 "function_type": ftype,
                                 "sym_visibility": "private"
                             },
                             regions=[region])

    @staticmethod
    def from_callable(name: str,
                      input_types: List[Attribute],
                      return_types: List[Attribute],
                      func: Block.BlockCallback,
                      private: Boolean = False):
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


COpT = TypeVar('COpT', bound='GenericCallOp')


@irdl_op_definition
class GenericCallOp(Operation):
    name: str = "toy.generic_call"
    arguments = VarOperandDef(AnyAttr())
    callee = AttributeDef(FlatSymbolRefAttr)

    # Note: naming this results triggers an ArgumentError
    res = VarResultDef(AnyAttr())
    # TODO how do we verify that the types are correct?

    @classmethod
    def get(cls: type[COpT], callee: Union[str, FlatSymbolRefAttr],
            operands: List[Union[SSAValue, Operation]],
            return_types: List[Attribute]) -> COpT:
        if isinstance(callee, str):
            callee = FlatSymbolRefAttr.from_str(callee)

        return super().create(operands=operands,
                              result_types=return_types,
                              attributes={"callee": callee})


MulOpT = TypeVar('MulOpT', bound='MulOp')


@irdl_op_definition
class MulOp(Operation):
    '''
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
    '''
    name: str = 'toy.mul'
    arguments = VarOperandDef(AnyAttr())
    result = ResultDef(AnyAttr())  # F64ElementsAttr

    @classmethod
    def from_summands(cls: type[MulOpT], lhs: OpResult,
                      rhs: SSAValue) -> MulOpT:
        return super().create(result_types=[lhs.typ], operands=[lhs, rhs])


PrintOpT = TypeVar('PrintOpT', bound='PrintOp')


@irdl_op_definition
class PrintOp(Operation):
    '''
    The "print" builtin operation prints a given input tensor, and produces
    no results.
    '''
    name: str = 'toy.print'
    arguments = VarOperandDef(AnyAttr())

    @classmethod
    def from_input(cls: type[PrintOpT], input: SSAValue) -> PrintOpT:
        return super().create(operands=[input])


ReturnOpT = TypeVar('ReturnOpT', bound='ReturnOp')


@irdl_op_definition
class ReturnOp(Operation):
    '''
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
    '''
    name: str = 'toy.return'
    arguments = OptOperandDef(AnyAttr())

    @classmethod
    def from_input(cls: type[ReturnOpT],
                   input: Optional[SSAValue] = None) -> ReturnOpT:
        return super().create(operands=[input] if input is not None else [])


ROpT = TypeVar('ROpT', bound='ReshapeOp')


@irdl_op_definition
class ReshapeOp(Operation):
    '''
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
    '''
    name: str = 'toy.reshape'
    arguments = VarOperandDef(AnyAttr())
    # We expect that the reshape operation returns a statically shaped tensor.
    result = ResultDef(AnyAttr())  # F64ElementsAttr

    @classmethod
    def from_input(cls: type[ROpT], input: SSAValue, shape: List[int]) -> ROpT:
        t = AnyTensorType.from_type_and_list(input.typ.element_type, shape)
        return super().create(result_types=[t], operands=[input])


@irdl_op_definition
class TransposeOp(Operation):
    name: str = 'toy.transpose'
    arguments = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())  # F64ElementsAttr

    @staticmethod
    def from_input(input: SSAValue):
        input_type = input.typ
        output_type: TensorType[Any] | UnrankedTensorType[Any]
        if isinstance(input_type, TensorType):
            output_type = TensorType.from_type_and_list(
                input_type.element_type, list(reversed(input_type.shape.data)))
        elif isinstance(input_type, UnrankedTensorType):
            output_type = input_type
        else:
            assert False, f'{input_type}: {type(input_type)}'

        return TransposeOp.create(operands=[input], result_types=[output_type])
