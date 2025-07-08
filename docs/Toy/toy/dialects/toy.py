"""
Toy language dialect from MLIR tutorial.
"""

from __future__ import annotations

from typing import TypeAlias, cast

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float64Type,
    FunctionType,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    UnrankedTensorType,
    f64,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    OpResult,
    OpTraits,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
)
from xdsl.traits import (
    CallableOpInterface,
    HasCanonicalizationPatternsTrait,
    HasShapeInferencePatternsTrait,
    IsTerminator,
    Pure,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

TensorTypeF64: TypeAlias = TensorType[Float64Type]
UnrankedTensorTypeF64: TypeAlias = UnrankedTensorType[Float64Type]
AnyTensorTypeF64: TypeAlias = TensorTypeF64 | UnrankedTensorTypeF64
AnyTensorTypeF64Constr = base(TensorTypeF64) | base(UnrankedTensorTypeF64)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
    """

    name = "toy.constant"
    value = attr_def(DenseIntOrFPElementsAttr[Float64Type])
    res = result_def(TensorTypeF64)

    traits = traits_def(Pure())

    def __init__(self, value: DenseIntOrFPElementsAttr):
        super().__init__(result_types=[value.type], attributes={"value": value})

    @staticmethod
    def from_list(data: list[float], shape: list[int]) -> ConstantOp:
        value = DenseIntOrFPElementsAttr.from_list(TensorType(f64, shape), data)
        return ConstantOp(value)

    @staticmethod
    def from_value(value: float) -> ConstantOp:
        return ConstantOp(
            DenseIntOrFPElementsAttr.from_list(TensorType(f64, []), (value,))
        )

    def verify_(self) -> None:
        if not self.res.type == self.value.type:
            raise VerifyException(
                "Expected value and result types to be equal: "
                f"{self.res.type}, {self.value.type}"
            )

    def get_type(self) -> TensorTypeF64:
        # Constant cannot be unranked
        return cast(TensorTypeF64, self.value.type)

    def get_shape(self) -> list[int]:
        return list(self.get_type().get_shape())

    def get_data(self) -> list[float]:
        return list(self.value.get_values())


class InferAddOpShapeInferencePattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, AddOp):
            return

        if not (
            isa(op_lhs_type := op.lhs.type, TensorType)
            and isinstance(op_rhs_type := op.rhs.type, TensorType)
        ):
            return
        assert op_lhs_type.get_shape() == op_rhs_type.get_shape()
        if isinstance(op_res_type := op.res.type, TensorType):
            assert op_lhs_type.get_shape() == op_res_type.get_shape()
        else:
            rewriter.replace_value_with_new_type(op.res, op_lhs_type)


class InferAddOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        return (InferAddOpShapeInferencePattern(),)


@irdl_op_definition
class AddOp(IRDLOperation):
    """
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
    """

    name = "toy.add"
    lhs = operand_def(AnyTensorTypeF64Constr)
    rhs = operand_def(AnyTensorTypeF64Constr)
    res = result_def(AnyTensorTypeF64Constr)

    traits = traits_def(Pure(), InferAddOpHasShapeInferencePatternsTrait())

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        if isa(lhs.type, TensorTypeF64):
            result_type = lhs.type
        else:
            result_type = rhs.type
        super().__init__(result_types=[result_type], operands=[lhs, rhs])

    def verify_(self):
        args = [self.lhs, self.rhs]

        shape = None
        for arg in args:
            # Expect shapes to be the same whenever they are defined, no check for unranked
            if isinstance(arg_type := arg.type, TensorType):
                if shape is None:
                    shape = arg_type.shape
                else:
                    if shape != arg_type.shape:
                        raise VerifyException(
                            "Expected AddOp args to have the same shape"
                        )


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body

    @classmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
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

    name = "toy.func"
    body = region_def()
    sym_name = attr_def(StringAttr)
    function_type = attr_def(FunctionType)
    sym_visibility = opt_attr_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), FuncOpCallableInterface())

    def __init__(
        self,
        name: str,
        ftype: FunctionType,
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        /,
        private: bool = False,
    ):
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": ftype,
        }
        if not isinstance(region, Region):
            region = Region(Block(arg_types=ftype.inputs))
        if private:
            attributes["sym_visibility"] = StringAttr("private")

        return super().__init__(attributes=attributes, regions=[region])

    def verify_(self):
        # Check that the returned value matches the type of the function
        if len(self.body.blocks) != 1:
            raise VerifyException("Expected FuncOp to contain one block")

        block = self.body.blocks[0]

        if not block.ops:
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.last_op

        if not isinstance(last_op, ReturnOp):
            raise VerifyException("Expected last op of FuncOp to be a ReturnOp")

        operand = last_op.input
        operand_type = None if operand is None else operand.type

        return_types = self.function_type.outputs.data

        if len(return_types):
            if len(return_types) == 1:
                return_type = return_types[0]
            else:
                raise VerifyException(
                    "Expected return type of func to have 0 or 1 values"
                )
        else:
            return_type = None

        if operand_type != return_type:
            raise VerifyException(
                "Expected return value to match return type of function"
            )


@irdl_op_definition
class GenericCallOp(IRDLOperation):
    name = "toy.generic_call"
    arguments = var_operand_def()
    callee = attr_def(SymbolRefAttr)

    # Note: naming this results triggers an ArgumentError
    res = var_result_def(AnyTensorTypeF64Constr)

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        operands: list[SSAValue | OpResult],
        return_types: list[Attribute],
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        return super().__init__(
            operands=[operands],
            result_types=[return_types],
            attributes={"callee": callee},
        )


class MulOpInferShapeInferencePattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, MulOp):
            return

        if not (
            isa(op_lhs_type := op.lhs.type, TensorType)
            and isinstance(op_rhs_type := op.rhs.type, TensorType)
        ):
            return

        assert op_lhs_type.get_shape() == op_rhs_type.get_shape()
        if isinstance(op_res_type := op.res.type, TensorType):
            assert op_lhs_type.get_shape() == op_res_type.get_shape()
        else:
            rewriter.replace_value_with_new_type(op.res, op_lhs_type)


class InferMulOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        return (MulOpInferShapeInferencePattern(),)


@irdl_op_definition
class MulOp(IRDLOperation):
    """
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
    """

    name = "toy.mul"
    lhs = operand_def(AnyTensorTypeF64Constr)
    rhs = operand_def(AnyTensorTypeF64Constr)
    res = result_def(AnyTensorTypeF64Constr)

    traits = traits_def(Pure(), InferMulOpHasShapeInferencePatternsTrait())

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        if isa(lhs.type, TensorTypeF64):
            result_type = lhs.type
        else:
            result_type = rhs.type
        super().__init__(result_types=[result_type], operands=[lhs, rhs])

    def verify_(self):
        args = [self.lhs, self.rhs]

        shape = None
        for arg in args:
            # Expect shapes to be the same whenever they are defined, no check for unranked
            if isinstance(arg_type := arg.type, TensorType):
                if shape is None:
                    shape = arg_type.shape
                else:
                    if shape != arg_type.shape:
                        raise VerifyException(
                            "Expected MulOp args to have the same shape"
                        )


@irdl_op_definition
class PrintOp(IRDLOperation):
    """
    The "print" builtin operation prints a given input tensor, and produces
    no results.
    """

    name = "toy.print"
    input = operand_def()

    def __init__(self, input: SSAValue):
        return super().__init__(operands=[input])


@irdl_op_definition
class ReturnOp(IRDLOperation):
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

    name = "toy.return"
    input = opt_operand_def(AnyTensorTypeF64Constr)

    traits = traits_def(IsTerminator())

    def __init__(self, input: SSAValue | None = None):
        return super().__init__(operands=[input])


class ReshapeOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from ..rewrites.optimise_toy import (
            FoldConstantReshapeOpPattern,
            ReshapeReshapeOpPattern,
        )

        return (ReshapeReshapeOpPattern(), FoldConstantReshapeOpPattern())


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    """
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
    """

    name = "toy.reshape"
    arg = operand_def(AnyTensorTypeF64Constr)
    # We expect that the reshape operation returns a statically shaped tensor.
    res = result_def(TensorTypeF64)

    traits = traits_def(Pure(), ReshapeOpHasCanonicalizationPatternsTrait())

    def __init__(self, arg: SSAValue, shape: list[int]):
        if not AnyTensorTypeF64Constr.verifies(arg.type):
            raise ValueError(
                f"Unexpected arg of type {arg.type} passed to ReshapeOp, expected"
                " {AnyTensorTypeF64}"
            )
        element_type = arg.type.element_type
        t = TensorTypeF64(element_type, shape)
        return super().__init__(result_types=[t], operands=[arg])

    @staticmethod
    def from_input_and_type(arg: SSAValue, t: TensorTypeF64) -> ReshapeOp:
        if not AnyTensorTypeF64Constr.verifies(arg.type):
            raise ValueError(
                f"Unexpected arg of type {arg.type} passed to ReshapeOp, expected"
                " {AnyTensorTypeF64}"
            )
        return ReshapeOp.create(result_types=[t], operands=[arg])

    def verify_(self):
        result_type = self.res.type
        assert isa(result_type, TensorTypeF64)
        if not len(result_type.shape.data):
            raise VerifyException("Reshape operation result shape should be defined")


class TransposeOpInferShapeInferencePattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, TransposeOp):
            return

        if not isinstance(op_arg_type := op.arg.type, TensorType):
            return

        arg_shape = op_arg_type.get_shape()
        res_shape = arg_shape[::-1]

        if isinstance(op_res_type := op.res.type, TensorType):
            assert res_shape == op_res_type.get_shape()
        else:
            rewriter.replace_value_with_new_type(op.res, TensorType(f64, res_shape))


class TransposeOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        return (TransposeOpInferShapeInferencePattern(),)


class TransposeOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from ..rewrites.optimise_toy import SimplifyRedundantTranspose

        return (SimplifyRedundantTranspose(),)


@irdl_op_definition
class TransposeOp(IRDLOperation):
    name = "toy.transpose"
    arg = operand_def(AnyTensorTypeF64Constr)
    res = result_def(AnyTensorTypeF64Constr)

    traits = OpTraits(
        lambda: (
            Pure(),
            TransposeOpHasShapeInferencePatternsTrait(),
            TransposeOpHasCanonicalizationPatternsTrait(),
        )
    )

    def __init__(self, arg: SSAValue):
        output_type: TensorTypeF64 | UnrankedTensorTypeF64
        if isa(arg.type, TensorTypeF64):
            element_type = arg.type.element_type
            output_type = TensorType(element_type, list(reversed(arg.type.get_shape())))
        else:
            if not isa(arg.type, UnrankedTensorTypeF64):
                raise ValueError(
                    f"Unexpected operand of type {arg.type} passed to TransposeOp, "
                    "expected {TensorTypeF64 | UnrankedTensorTypeF64}"
                )
            output_type = arg.type

        super().__init__(operands=[arg], result_types=[output_type])


class CastOpInferShapeInferencePattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, CastOp):
            return

        if not isinstance(op_arg_type := op.arg.type, TensorType):
            return

        shape = op_arg_type.get_shape()

        if isinstance(op_res_type := op.res.type, TensorType):
            assert shape == op_res_type.get_shape()
            rewriter.replace_matched_op((), (op.arg,))
        else:
            rewriter.replace_value_with_new_type(op.res, TensorType(f64, shape))


class CastOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        return (CastOpInferShapeInferencePattern(),)


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "toy.cast"
    arg = operand_def(AnyTensorTypeF64Constr)
    res = result_def(AnyTensorTypeF64Constr)

    traits = traits_def(Pure(), CastOpHasShapeInferencePatternsTrait())

    def __init__(self, arg: SSAValue, res: AnyTensorTypeF64 | None = None):
        if res is None:
            res = UnrankedTensorType(f64)

        return super().__init__(
            operands=[arg],
            result_types=[res],
        )


Toy = Dialect(
    "toy",
    [
        ConstantOp,
        AddOp,
        FuncOp,
        GenericCallOp,
        PrintOp,
        MulOp,
        ReturnOp,
        ReshapeOp,
        TransposeOp,
        CastOp,
    ],
    [],
)
