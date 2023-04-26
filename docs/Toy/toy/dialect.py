"""
Toy language dialect from MLIR tutorial.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias, cast

from xdsl.ir import (Dialect, SSAValue, Attribute, Block, Region, Operation,
                     OpResult)
from xdsl.dialects.builtin import (Float64Type, FunctionType, SymbolRefAttr,
                                   TensorType, UnrankedTensorType, f64,
                                   DenseIntOrFPElementsAttr, StringAttr)
from xdsl.irdl import (OpAttr, Operand, OptOpAttr, OptOperand, VarOpResult,
                       VarOperand, irdl_op_definition, AnyAttr)

from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

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
    def from_list(data: list[float], shape: list[int]) -> ConstantOp:
        value = DenseIntOrFPElementsAttr.tensor_from_list(data, f64, shape)
        return ConstantOp.from_value(value)

    @staticmethod
    def from_value(value: DenseIntOrFPElementsAttr) -> ConstantOp:
        return ConstantOp.create(result_types=[value.type],
                                 attributes={"value": value})

    def verify_(self) -> None:
        if not self.res.typ == self.value.type:
            raise VerifyException(
                "Expected value and result types to be equal: "
                f"{self.res.typ}, {self.value.type}")

    def get_type(self) -> TensorTypeF64:
        # Constant cannot be unranked
        return cast(TensorTypeF64, self.value.type)

    def get_shape(self) -> list[int]:
        return self.get_type().get_shape()

    def get_data(self) -> list[float]:
        return [float(el.value.data) for el in self.value.data.data]


@irdl_op_definition
class AddOp(Operation):
    """
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
    """
    name: str = 'toy.add'
    lhs: Annotated[Operand, AnyTensorTypeF64]
    rhs: Annotated[Operand, AnyTensorTypeF64]
    res: Annotated[OpResult, AnyTensorTypeF64]

    @staticmethod
    def from_summands(lhs: SSAValue, rhs: SSAValue) -> AddOp:
        if isa(lhs.typ, TensorTypeF64):
            result_typ = lhs.typ
        else:
            result_typ = rhs.typ
        return AddOp.create(result_types=[result_typ], operands=[lhs, rhs])

    def verify_(self):
        args = [self.lhs, self.rhs]

        shape = None
        for arg in args:
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
    name: str = 'toy.func'
    body: Region
    sym_name: OpAttr[StringAttr]
    function_type: OpAttr[FunctionType]
    sym_visibility: OptOpAttr[StringAttr]

    @staticmethod
    def from_region(name: str,
                    ftype: FunctionType,
                    region: Region,
                    /,
                    private: bool = False):
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": ftype,
        }
        if private:
            attributes["sym_visibility"] = StringAttr("private")

        return FuncOp.create(attributes=attributes, regions=[region])

    @staticmethod
    def from_callable(name: str,
                      input_types: list[Attribute],
                      return_types: list[Attribute],
                      func: Block.BlockCallback,
                      /,
                      private: bool = False):
        ftype = FunctionType.from_lists(input_types, return_types)
        return FuncOp.from_region(
            name,
            ftype,
            Region.from_block_list([Block.from_callable(input_types, func)]),
            private=private)

    def verify_(self):
        # Check that the returned value matches the type of the function
        if len(self.body.blocks) != 1:
            raise VerifyException("Expected FuncOp to contain one block")

        block = self.body.blocks[0]

        if not len(block.ops):
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.ops[-1]

        if not isinstance(last_op, ReturnOp):
            raise VerifyException(
                "Expected last op of FuncOp to be a ReturnOp")

        operand = last_op.input
        operand_typ = None if operand is None else operand.typ

        return_typs = self.function_type.outputs.data

        if len(return_typs):
            if len(return_typs) == 1:
                return_typ = return_typs[0]
            else:
                raise VerifyException(
                    "Expected return type of func to have 0 or 1 values")
        else:
            return_typ = None

        if operand_typ != return_typ:
            raise VerifyException(
                "Expected return value to match return type of function")


@irdl_op_definition
class GenericCallOp(Operation):
    name: str = "toy.generic_call"
    arguments: Annotated[VarOperand, AnyAttr()]
    callee: OpAttr[SymbolRefAttr]

    # Note: naming this results triggers an ArgumentError
    res: Annotated[VarOpResult, AnyTensorTypeF64]

    @staticmethod
    def get(callee: str | SymbolRefAttr, operands: list[SSAValue | OpResult],
            return_types: list[Attribute]) -> GenericCallOp:
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        return GenericCallOp.create(operands=operands,
                                    result_types=return_types,
                                    attributes={"callee": callee})


@irdl_op_definition
class MulOp(Operation):
    """
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
    """
    name: str = 'toy.mul'
    lhs: Annotated[Operand, AnyTensorTypeF64]
    rhs: Annotated[Operand, AnyTensorTypeF64]
    res: Annotated[OpResult, AnyTensorTypeF64]

    @staticmethod
    def from_summands(lhs: SSAValue, rhs: SSAValue) -> MulOp:
        if isa(lhs.typ, TensorTypeF64):
            result_typ = lhs.typ
        else:
            result_typ = rhs.typ
        return MulOp.create(result_types=[result_typ], operands=[lhs, rhs])

    def verify_(self):
        args = [self.lhs, self.rhs]

        shape = None
        for arg in args:
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
    input: Annotated[Operand, AnyAttr()]

    @staticmethod
    def from_input(input: SSAValue) -> PrintOp:
        return PrintOp.create(operands=[input])


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
    input: Annotated[OptOperand, AnyTensorTypeF64]

    @staticmethod
    def from_input(input: SSAValue | None = None) -> ReturnOp:
        return ReturnOp.create(operands=[input] if input is not None else [])


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
    arg: Annotated[Operand, AnyTensorTypeF64]
    # We expect that the reshape operation returns a statically shaped tensor.
    res: Annotated[OpResult, TensorTypeF64]

    @staticmethod
    def from_input(arg: SSAValue, shape: list[int]) -> ReshapeOp:
        if not isa(arg.typ, AnyTensorTypeF64):
            raise ValueError(
                f'Unexpected arg of type {arg.typ} passed to ReshapeOp, expected {AnyTensorTypeF64}'
            )
        element_type = arg.typ.element_type
        t = TensorTypeF64.from_type_and_list(element_type, shape)
        return ReshapeOp.create(result_types=[t], operands=[arg])

    @staticmethod
    def from_input_and_type(arg: SSAValue, t: TensorTypeF64) -> ReshapeOp:
        if not isa(arg.typ, AnyTensorTypeF64):
            raise ValueError(
                f'Unexpected arg of type {arg.typ} passed to ReshapeOp, expected {AnyTensorTypeF64}'
            )
        return ReshapeOp.create(result_types=[t], operands=[arg])

    def verify_(self):
        result_type = self.res.typ
        assert isa(result_type, TensorTypeF64)
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
        output_type: TensorTypeF64 | UnrankedTensorTypeF64
        if isa(input.typ, TensorTypeF64):
            element_type = input.typ.element_type
            output_type = TensorType.from_type_and_list(
                element_type, list(reversed(input.typ.get_shape())))
        else:
            if not isa(input.typ, UnrankedTensorTypeF64):
                raise ValueError(
                    f'Unexpected arg of type {input.typ} passed to TransposeOp, expected {TensorTypeF64 | UnrankedTensorTypeF64}'
                )
            output_type = input.typ

        return TransposeOp.create(operands=[input], result_types=[output_type])


Toy = Dialect([
    ConstantOp,
    AddOp,
    FuncOp,
    GenericCallOp,
    PrintOp,
    MulOp,
    ReturnOp,
    ReshapeOp,
    TransposeOp,
], [])
