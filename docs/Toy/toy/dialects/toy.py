"""
Toy language dialect from MLIR tutorial.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float64Type,
    FunctionType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    TensorType,
    UnrankedTensorType,
    f64,
)
from xdsl.interfaces import HasCanonicalizationPatternsInterface
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
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
    op_type_rewrite_pattern,
)
from xdsl.traits import (
    CallableOpInterface,
    HasParent,
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
    lhs = operand_def(AnyTensorTypeF64)
    rhs = operand_def(AnyTensorTypeF64)
    res = result_def(AnyTensorTypeF64)

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
    body = region_def("single_block")
    sym_name = attr_def(SymbolNameConstraint())
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
        block = self.body.block

        if not block.ops:
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.last_op

        if not isinstance(last_op, ReturnOp):
            raise VerifyException("Expected last op of FuncOp to be a ReturnOp")


@irdl_op_definition
class GenericCallOp(IRDLOperation):
    name = "toy.generic_call"
    arguments = var_operand_def()
    callee = attr_def(SymbolRefAttr)

    res = var_result_def(AnyTensorTypeF64)

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        operands: Sequence[SSAValue],
        return_types: Sequence[Attribute],
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
    lhs = operand_def(AnyTensorTypeF64)
    rhs = operand_def(AnyTensorTypeF64)
    res = result_def(AnyTensorTypeF64)

    traits = traits_def(Pure(), InferMulOpHasShapeInferencePatternsTrait())

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[lhs.type], operands=[lhs, rhs])

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
    input = opt_operand_def(AnyTensorTypeF64)

    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, input: SSAValue | None = None):
        return super().__init__(operands=[input])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        function_return_types = func_op.function_type.outputs.data
        if function_return_types != tuple(self.operand_types):
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )


@irdl_op_definition
class ReshapeOp(HasCanonicalizationPatternsInterface, IRDLOperation):
    """
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
    """

    name = "toy.reshape"
    arg = operand_def(AnyTensorTypeF64)
    res = result_def(TensorTypeF64)

    traits = traits_def(Pure())

    def __init__(self, arg: SSAValue, result_type: TensorTypeF64 | Sequence[int]):
        if not isinstance(result_type, TensorType):
            result_type = TensorType(f64, result_type)
        return super().__init__(result_types=[result_type], operands=[arg])

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (ReshapeReshapeOpPattern(), FoldConstantReshapeOpPattern())


class ReshapeReshapeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter):
        """
        Reshape(Reshape(x)) = Reshape(x)
        """
        if isinstance(op.arg.owner, ReshapeOp):
            rewriter.replace_op(op, ReshapeOp(op.arg.owner.arg, op.res.type))


class FoldConstantReshapeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter):
        """
        Reshaping a constant can be done at compile time
        """
        if isinstance(op.arg.owner, ConstantOp):
            rewriter.replace_op(
                op,
                ConstantOp(
                    DenseIntOrFPElementsAttr.from_list(
                        type=op.res.type, data=op.arg.owner.value.get_values()
                    )
                ),
            )


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


@irdl_op_definition
class TransposeOp(HasCanonicalizationPatternsInterface, IRDLOperation):
    name = "toy.transpose"
    arg = operand_def(AnyTensorTypeF64)
    res = result_def(AnyTensorTypeF64)

    traits = traits_def(Pure(), TransposeOpHasShapeInferencePatternsTrait())

    def __init__(self, arg: SSAValue):
        if isa(arg.type, TensorTypeF64):
            output_type = TensorType(
                arg.type.element_type, reversed(arg.type.get_shape())
            )
        else:
            output_type = arg.type

        super().__init__(operands=[arg], result_types=[output_type])

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (SimplifyRedundantTranspose(),)


class SimplifyRedundantTranspose(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter):
        """
        Fold transpose(transpose(x)) -> x
        """
        if isinstance(op.arg.owner, TransposeOp):
            rewriter.replace_op(op, [], [op.arg.owner.arg])


class CastOpInferShapeInferencePattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, CastOp):
            return

        if not isinstance(op_arg_type := op.arg.type, TensorType):
            return

        shape = op_arg_type.get_shape()

        if isinstance(op_res_type := op.res.type, TensorType):
            assert shape == op_res_type.get_shape()
            rewriter.replace_op(op, (), (op.arg,))
        else:
            rewriter.replace_value_with_new_type(op.res, TensorType(f64, shape))


class CastOpHasShapeInferencePatternsTrait(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls) -> tuple[RewritePattern, ...]:
        return (CastOpInferShapeInferencePattern(),)


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "toy.cast"
    arg = operand_def(AnyTensorTypeF64)
    res = result_def(AnyTensorTypeF64)

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
