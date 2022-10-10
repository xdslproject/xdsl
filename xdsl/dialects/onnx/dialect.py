# This file was generated using the script src/tools/tablegen_to_irdl.py. Editing it is a bad idea.
from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *


@dataclass
class Onnx:
    ctx: MLContext

    def __post_init__(self):
        # From AdditionalONNXOps.td
        self.ctx.register_op(ONNXONNX_CallOp)
        self.ctx.register_op(ONNXCustomOp)
        self.ctx.register_op(ONNXDimOp)
        self.ctx.register_op(ONNXEntryPointOp)
        self.ctx.register_op(ONNXPrintSignatureOp)
        self.ctx.register_op(ONNXMaxPoolSingleOutOp)

        # From ONNXOps.td.inc
        self.ctx.register_op(ONNXAbsOp)
        self.ctx.register_op(ONNXAcosOp)
        self.ctx.register_op(ONNXAcoshOp)
        self.ctx.register_op(ONNXAddOp)
        self.ctx.register_op(ONNXAndOp)
        self.ctx.register_op(ONNXArgMaxOp)
        self.ctx.register_op(ONNXArgMinOp)
        self.ctx.register_op(ONNXAsinOp)
        self.ctx.register_op(ONNXAsinhOp)
        self.ctx.register_op(ONNXAtanOp)
        self.ctx.register_op(ONNXAtanhOp)
        self.ctx.register_op(ONNXAveragePoolOp)
        self.ctx.register_op(ONNXBatchNormalizationOp)
        self.ctx.register_op(ONNXBernoulliOp)
        self.ctx.register_op(ONNXBitShiftOp)
        self.ctx.register_op(ONNXCastOp)
        self.ctx.register_op(ONNXCastLikeOp)
        self.ctx.register_op(ONNXCeilOp)
        self.ctx.register_op(ONNXCeluOp)
        self.ctx.register_op(ONNXClipOp)
        self.ctx.register_op(ONNXClipV12Op)
        self.ctx.register_op(ONNXClipV11Op)
        self.ctx.register_op(ONNXClipV6Op)
        self.ctx.register_op(ONNXCompressOp)
        self.ctx.register_op(ONNXConcatOp)
        self.ctx.register_op(ONNXConcatFromSequenceOp)
        self.ctx.register_op(ONNXConstantOp)
        self.ctx.register_op(ONNXConstantOfShapeOp)
        self.ctx.register_op(ONNXConvOp)
        self.ctx.register_op(ONNXConvIntegerOp)
        self.ctx.register_op(ONNXConvTransposeOp)
        self.ctx.register_op(ONNXCosOp)
        self.ctx.register_op(ONNXCoshOp)
        self.ctx.register_op(ONNXCumSumOp)
        self.ctx.register_op(ONNXDepthToSpaceOp)
        self.ctx.register_op(ONNXDequantizeLinearOp)
        self.ctx.register_op(ONNXDetOp)
        self.ctx.register_op(ONNXDivOp)
        self.ctx.register_op(ONNXDropoutOp)
        self.ctx.register_op(ONNXDynamicQuantizeLinearOp)
        self.ctx.register_op(ONNXEinsumOp)
        self.ctx.register_op(ONNXEluOp)
        self.ctx.register_op(ONNXEqualOp)
        self.ctx.register_op(ONNXErfOp)
        self.ctx.register_op(ONNXExpOp)
        self.ctx.register_op(ONNXExpandOp)
        self.ctx.register_op(ONNXEyeLikeOp)
        self.ctx.register_op(ONNXFlattenOp)
        self.ctx.register_op(ONNXFloorOp)
        self.ctx.register_op(ONNXGRUOp)
        self.ctx.register_op(ONNXGatherOp)
        self.ctx.register_op(ONNXGatherElementsOp)
        self.ctx.register_op(ONNXGatherNDOp)
        self.ctx.register_op(ONNXGemmOp)
        self.ctx.register_op(ONNXGlobalAveragePoolOp)
        self.ctx.register_op(ONNXGlobalLpPoolOp)
        self.ctx.register_op(ONNXGlobalMaxPoolOp)
        self.ctx.register_op(ONNXGreaterOp)
        self.ctx.register_op(ONNXGreaterOrEqualOp)
        self.ctx.register_op(ONNXGridSampleOp)
        self.ctx.register_op(ONNXHardSigmoidOp)
        self.ctx.register_op(ONNXHardSwishOp)
        self.ctx.register_op(ONNXHardmaxOp)
        self.ctx.register_op(ONNXIdentityOp)
        self.ctx.register_op(ONNXIfOp)
        self.ctx.register_op(ONNXInstanceNormalizationOp)
        self.ctx.register_op(ONNXIsInfOp)
        self.ctx.register_op(ONNXIsNaNOp)
        self.ctx.register_op(ONNXLRNOp)
        self.ctx.register_op(ONNXLSTMOp)
        self.ctx.register_op(ONNXLeakyReluOp)
        self.ctx.register_op(ONNXLessOp)
        self.ctx.register_op(ONNXLessOrEqualOp)
        self.ctx.register_op(ONNXLogOp)
        self.ctx.register_op(ONNXLogSoftmaxOp)
        self.ctx.register_op(ONNXLoopOp)
        self.ctx.register_op(ONNXLpNormalizationOp)
        self.ctx.register_op(ONNXLpPoolOp)
        self.ctx.register_op(ONNXMatMulOp)
        self.ctx.register_op(ONNXMatMulIntegerOp)
        self.ctx.register_op(ONNXMaxOp)
        self.ctx.register_op(ONNXMaxPoolOp)
        self.ctx.register_op(ONNXMaxRoiPoolOp)
        self.ctx.register_op(ONNXMaxUnpoolOp)
        self.ctx.register_op(ONNXMeanOp)
        self.ctx.register_op(ONNXMeanVarianceNormalizationOp)
        self.ctx.register_op(ONNXMinOp)
        self.ctx.register_op(ONNXModOp)
        self.ctx.register_op(ONNXMulOp)
        self.ctx.register_op(ONNXMultinomialOp)
        self.ctx.register_op(ONNXNegOp)
        self.ctx.register_op(ONNXNegativeLogLikelihoodLossOp)
        self.ctx.register_op(ONNXNonMaxSuppressionOp)
        self.ctx.register_op(ONNXNonZeroOp)
        self.ctx.register_op(ONNXNotOp)
        self.ctx.register_op(ONNXOneHotOp)
        self.ctx.register_op(ONNXOptionalOp)
        self.ctx.register_op(ONNXOptionalGetElementOp)
        self.ctx.register_op(ONNXOptionalHasElementOp)
        self.ctx.register_op(ONNXOrOp)
        self.ctx.register_op(ONNXPReluOp)
        self.ctx.register_op(ONNXPadOp)
        self.ctx.register_op(ONNXPadV11Op)
        self.ctx.register_op(ONNXPadV2Op)
        self.ctx.register_op(ONNXPowOp)
        self.ctx.register_op(ONNXQLinearConvOp)
        self.ctx.register_op(ONNXQLinearMatMulOp)
        self.ctx.register_op(ONNXQuantizeLinearOp)
        self.ctx.register_op(ONNXRNNOp)
        self.ctx.register_op(ONNXRandomNormalOp)
        self.ctx.register_op(ONNXRandomNormalLikeOp)
        self.ctx.register_op(ONNXRandomUniformOp)
        self.ctx.register_op(ONNXRandomUniformLikeOp)
        self.ctx.register_op(ONNXRangeOp)
        self.ctx.register_op(ONNXReciprocalOp)
        self.ctx.register_op(ONNXReduceL1Op)
        self.ctx.register_op(ONNXReduceL2Op)
        self.ctx.register_op(ONNXReduceLogSumOp)
        self.ctx.register_op(ONNXReduceLogSumExpOp)
        self.ctx.register_op(ONNXReduceMaxOp)
        self.ctx.register_op(ONNXReduceMeanOp)
        self.ctx.register_op(ONNXReduceMinOp)
        self.ctx.register_op(ONNXReduceProdOp)
        self.ctx.register_op(ONNXReduceSumOp)
        self.ctx.register_op(ONNXReduceSumV11Op)
        self.ctx.register_op(ONNXReduceSumSquareOp)
        self.ctx.register_op(ONNXReluOp)
        self.ctx.register_op(ONNXReshapeOp)
        self.ctx.register_op(ONNXResizeOp)
        self.ctx.register_op(ONNXResizeV11Op)
        self.ctx.register_op(ONNXResizeV10Op)
        self.ctx.register_op(ONNXReverseSequenceOp)
        self.ctx.register_op(ONNXRoiAlignOp)
        self.ctx.register_op(ONNXRoundOp)
        self.ctx.register_op(ONNXScanOp)
        self.ctx.register_op(ONNXScatterOp)
        self.ctx.register_op(ONNXScatterElementsOp)
        self.ctx.register_op(ONNXScatterNDOp)
        self.ctx.register_op(ONNXSeluOp)
        self.ctx.register_op(ONNXSequenceAtOp)
        self.ctx.register_op(ONNXSequenceConstructOp)
        self.ctx.register_op(ONNXSequenceEmptyOp)
        self.ctx.register_op(ONNXSequenceEraseOp)
        self.ctx.register_op(ONNXSequenceInsertOp)
        self.ctx.register_op(ONNXSequenceLengthOp)
        self.ctx.register_op(ONNXShapeOp)
        self.ctx.register_op(ONNXShrinkOp)
        self.ctx.register_op(ONNXSigmoidOp)
        self.ctx.register_op(ONNXSignOp)
        self.ctx.register_op(ONNXSinOp)
        self.ctx.register_op(ONNXSinhOp)
        self.ctx.register_op(ONNXSizeOp)
        self.ctx.register_op(ONNXSliceOp)
        self.ctx.register_op(ONNXSoftmaxOp)
        self.ctx.register_op(ONNXSoftmaxCrossEntropyLossOp)
        self.ctx.register_op(ONNXSoftplusOp)
        self.ctx.register_op(ONNXSoftsignOp)
        self.ctx.register_op(ONNXSpaceToDepthOp)
        self.ctx.register_op(ONNXSplitOp)
        self.ctx.register_op(ONNXSplitV11Op)
        self.ctx.register_op(ONNXSplitToSequenceOp)
        self.ctx.register_op(ONNXSqrtOp)
        self.ctx.register_op(ONNXSqueezeOp)
        self.ctx.register_op(ONNXSqueezeV11Op)
        self.ctx.register_op(ONNXStringNormalizerOp)
        self.ctx.register_op(ONNXSubOp)
        self.ctx.register_op(ONNXSumOp)
        self.ctx.register_op(ONNXTanOp)
        self.ctx.register_op(ONNXTanhOp)
        self.ctx.register_op(ONNXTfIdfVectorizerOp)
        self.ctx.register_op(ONNXThresholdedReluOp)
        self.ctx.register_op(ONNXTileOp)
        self.ctx.register_op(ONNXTopKOp)
        self.ctx.register_op(ONNXTransposeOp)
        self.ctx.register_op(ONNXTriluOp)
        self.ctx.register_op(ONNXUniqueOp)
        self.ctx.register_op(ONNXUnsqueezeOp)
        self.ctx.register_op(ONNXUnsqueezeV11Op)
        self.ctx.register_op(ONNXUpsampleOp)
        self.ctx.register_op(ONNXUpsampleV9Op)
        self.ctx.register_op(ONNXUpsampleV7Op)
        self.ctx.register_op(ONNXWhereOp)
        self.ctx.register_op(ONNXXorOp)
        self.ctx.register_op(ONNXArrayFeatureExtractorOp)
        self.ctx.register_op(ONNXBinarizerOp)
        self.ctx.register_op(ONNXCastMapOp)
        self.ctx.register_op(ONNXCategoryMapperOp)
        self.ctx.register_op(ONNXDictVectorizerOp)
        self.ctx.register_op(ONNXFeatureVectorizerOp)
        self.ctx.register_op(ONNXImputerOp)
        self.ctx.register_op(ONNXLabelEncoderOp)
        self.ctx.register_op(ONNXLinearClassifierOp)
        self.ctx.register_op(ONNXLinearRegressorOp)
        self.ctx.register_op(ONNXNormalizerOp)
        self.ctx.register_op(ONNXOneHotEncoderOp)
        self.ctx.register_op(ONNXSVMClassifierOp)
        self.ctx.register_op(ONNXSVMRegressorOp)
        self.ctx.register_op(ONNXScalerOp)
        self.ctx.register_op(ONNXTreeEnsembleClassifierOp)
        self.ctx.register_op(ONNXTreeEnsembleRegressorOp)
        self.ctx.register_op(ONNXZipMapOp)
        self.ctx.register_op(ONNXAdagradOp)
        self.ctx.register_op(ONNXAdamOp)
        self.ctx.register_op(ONNXGradientOp)
        self.ctx.register_op(ONNXMomentumOp)


@irdl_op_definition
class ONNXONNX_CallOp(Operation):
    name: str = "onnx.ONNX_Call"
    summary: str = \
r"""call operation"""
    description: str =\
r"""
The `call` operation represents a direct call to a function that is within
the same symbol scope as the call. The operands and result types of the
call must match the specified function type. The callee is encoded as a
symbol reference attribute named callee.
"""
    callee = OptAttributeDef(Attribute)  # should actually be FlatSymbolRefAttr
    operands = VarOperandDef(Attribute)  # should actually be Variadic<AnyType>
    result = VarResultDef(
        Attribute)  # should actually be outs Variadic<AnyTypeOf<[AnyTensor]>>


@irdl_op_definition
class ONNXCustomOp(Operation):
    name: str = "onnx.Custom"
    summary: str = \
r"""ONNX Custom operation"""
    description: str =\
r"""
Allow call-out to a user defined operation. A single attribute
is a string which names the operation, other inputs are
passed to the user operation.
The number of inputs and outputs can vary.
"""
    input = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>
    function_name = OptAttributeDef(Attribute)  # should actually be StrAttr
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>


@irdl_op_definition
class ONNXDimOp(Operation):
    name: str = "onnx.Dim"
    summary: str = \
r"""ONNX dimensions operation."""
    description: str =\
r"""
This operation is to obtain the dimension of a Tensor"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    dim = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXEntryPointOp(Operation):
    name: str = "onnx.EntryPoint"
    summary: str = \
r"""Indicate ONNX entry point"""
    description: str =\
r"""
The onnx.EntryPoint function indicates the main entry point of ONNX model.
"""
    func = OptAttributeDef(Attribute)  # should actually be SymbolRefAttr


@irdl_op_definition
class ONNXPrintSignatureOp(Operation):
    name: str = "onnx.PrintSignature"
    summary: str = \
r"""ONNX Op to print type signature of its input operands"""
    description: str =\
r"""
Print type signature of the op's input operands. This operation is introduced early
so as to preserve the name of the original ONNX op.
"""
    op_name = OptAttributeDef(Attribute)  # should actually be StrAttr
    input = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[AnyTensor, NoneType]>>
    o_Y = VarResultDef(
        Attribute)  # should actually be outs AnyTypeOf<[AnyMemRef, AnyTensor]>


@irdl_op_definition
class ONNXMaxPoolSingleOutOp(Operation):
    name: str = "onnx.MaxPoolSingleOut"
    summary: str = \
r"""ONNX MaxPool operation with a single output."""
    description: str =\
r"""
ONNX MaxPool operation with a single output.
See ONNXMaxPoolOp for a full description of the MaxPool semantics.
"""
    X = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[AnyMemRef, AnyTensor]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    ceil_mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    dilations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<I64ArrayAttr, "{}">
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    storage_order = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    o_Y = ResultDef(
        Attribute)  # should actually be outs AnyTypeOf<[AnyMemRef, AnyTensor]>


@irdl_op_definition
class ONNXAbsOp(Operation):
    name: str = "onnx.Abs"
    summary: str = \
r"""ONNX Abs operation"""
    description: str =\
r"""
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXAcosOp(Operation):
    name: str = "onnx.Acos"
    summary: str = \
r"""ONNX Acos operation"""
    description: str =\
r"""
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXAcoshOp(Operation):
    name: str = "onnx.Acosh"
    summary: str = \
r"""ONNX Acosh operation"""
    description: str =\
r"""
Calculates the hyperbolic arccosine of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXAddOp(Operation):
    name: str = "onnx.Add"
    summary: str = \
r"""ONNX Add operation"""
    description: str =\
r"""
Performs element-wise binary addition (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXAndOp(Operation):
    name: str = "onnx.And"
    summary: str = \
r"""ONNX And operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    B = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXArgMaxOp(Operation):
    name: str = "onnx.ArgMax"
    summary: str = \
r"""ONNX ArgMax operation"""
    description: str =\
r"""
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulting tensor have the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    select_last_index = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    reduced = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXArgMinOp(Operation):
    name: str = "onnx.ArgMin"
    summary: str = \
r"""ONNX ArgMin operation"""
    description: str =\
r"""
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulting tensor have the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    select_last_index = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    reduced = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXAsinOp(Operation):
    name: str = "onnx.Asin"
    summary: str = \
r"""ONNX Asin operation"""
    description: str =\
r"""
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXAsinhOp(Operation):
    name: str = "onnx.Asinh"
    summary: str = \
r"""ONNX Asinh operation"""
    description: str =\
r"""
Calculates the hyperbolic arcsine of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXAtanOp(Operation):
    name: str = "onnx.Atan"
    summary: str = \
r"""ONNX Atan operation"""
    description: str =\
r"""
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXAtanhOp(Operation):
    name: str = "onnx.Atanh"
    summary: str = \
r"""ONNX Atanh operation"""
    description: str =\
r"""
Calculates the hyperbolic arctangent of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXAveragePoolOp(Operation):
    name: str = "onnx.AveragePool"
    summary: str = \
r"""ONNX AveragePool operation"""
    description: str =\
r"""
AveragePool consumes an input tensor X and applies average pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
average pooling consisting of computing the average on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shapei = floor((input_spatial_shapei + pad_shapei - kernel_spatial_shapei) / strides_spatial_shapei + 1)
```
or
```
output_spatial_shapei = ceil((input_spatial_shapei + pad_shapei - kernel_spatial_shapei) / strides_spatial_shapei + 1)
```
if ceil_mode is enabled
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    ceil_mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    count_include_pad = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXBatchNormalizationOp(Operation):
    name: str = "onnx.BatchNormalization"
    summary: str = \
r"""ONNX BatchNormalization operation"""
    description: str =\
r"""
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    scale = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    input_mean = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    input_var = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    epsilon = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1e-05">
    momentum = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.9">
    training_mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    running_mean = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>
    running_var = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>


@irdl_op_definition
class ONNXBernoulliOp(Operation):
    name: str = "onnx.Bernoulli"
    summary: str = \
r"""ONNX Bernoulli operation"""
    description: str =\
r"""
Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range 0,1) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    dtype = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[I1]>]>


@irdl_op_definition
class ONNXBitShiftOp(Operation):
    name: str = "onnx.BitShift"
    summary: str = \
r"""ONNX BitShift operation"""
    description: str =\
r"""
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute \direction\ is \RIGHT\, this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute \direction\
is \LEFT\, bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if \direction\ is \Right\, X is 1, 4,
and S is 1, 1, the corresponding output Z would be 0, 2. If \direction\ is \LEFT\ with
X=1, 2 and S=1, 2, the corresponding output Y would be 2, 8.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>
    Y = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>
    direction = OptAttributeDef(Attribute)  # should actually be StrAttr
    Z = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>


@irdl_op_definition
class ONNXCastOp(Operation):
    name: str = "onnx.Cast"
    summary: str = \
r"""ONNX Cast operation"""
    description: str =\
r"""
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>
    to = OptAttributeDef(Attribute)  # should actually be TypeAttr
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXCastLikeOp(Operation):
    name: str = "onnx.CastLike"
    summary: str = \
r"""ONNX CastLike operation"""
    description: str =\
r"""
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>
    target_type = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXCeilOp(Operation):
    name: str = "onnx.Ceil"
    summary: str = \
r"""ONNX Ceil operation"""
    description: str =\
r"""
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXCeluOp(Operation):
    name: str = "onnx.Celu"
    summary: str = \
r"""ONNX Celu operation"""
    description: str =\
r"""
Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:
"""
    X = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXClipOp(Operation):
    name: str = "onnx.Clip"
    summary: str = \
r"""ONNX Clip operation"""
    description: str =\
r"""
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    min = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>
    max = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXClipV12Op(Operation):
    name: str = "onnx.ClipV12"
    summary: str = \
r"""ONNX Clip operation"""
    description: str =\
r"""
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    min = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    max = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXClipV11Op(Operation):
    name: str = "onnx.ClipV11"
    summary: str = \
r"""ONNX Clip operation"""
    description: str =\
r"""
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    min = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    max = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXClipV6Op(Operation):
    name: str = "onnx.ClipV6"
    summary: str = \
r"""ONNX Clip operation"""
    description: str =\
r"""
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    max = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedAttr<F32Attr, "(3.402823e+38)">
    min = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedAttr<F32Attr, "(-3.402823e+38)">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXCompressOp(Operation):
    name: str = "onnx.Compress"
    summary: str = \
r"""ONNX Compress operation"""
    description: str =\
r"""
Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
In case axis is not provided, input is flattened before elements are selected.
Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    condition = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    axis = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXConcatOp(Operation):
    name: str = "onnx.Concat"
    summary: str = \
r"""ONNX Concat operation"""
    description: str =\
r"""
Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
"""
    inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>
    axis = OptAttributeDef(Attribute)  # should actually be SI64Attr
    concat_result = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXConcatFromSequenceOp(Operation):
    name: str = "onnx.ConcatFromSequence"
    summary: str = \
r"""ONNX ConcatFromSequence operation"""
    description: str =\
r"""
Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
"""
    input_sequence = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>
    axis = OptAttributeDef(Attribute)  # should actually be SI64Attr
    new_axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    concat_result = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXConstantOp(Operation):
    name: str = "onnx.Constant"
    summary: str = \
r"""ONNX Constant operation"""
    description: str =\
r"""
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
"""
    sparse_value = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<AnyAttr>
    value = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<AnyAttr>
    value_float = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    value_floats = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    value_int = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    value_ints = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    value_string = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrAttr>
    value_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXConstantOfShapeOp(Operation):
    name: str = "onnx.ConstantOfShape"
    summary: str = \
r"""ONNX ConstantOfShape operation"""
    description: str =\
r"""
Generate a tensor with given value and shape.
"""
    input = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    value = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<AnyAttr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>


@irdl_op_definition
class ONNXConvOp(Operation):
    name: str = "onnx.Conv"
    summary: str = \
r"""ONNX Conv operation"""
    description: str =\
r"""
The convolution operator consumes an input tensor and a filter, and
computes the output.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    W = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    dilations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    group = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXConvIntegerOp(Operation):
    name: str = "onnx.ConvInteger"
    summary: str = \
r"""ONNX ConvInteger operation"""
    description: str =\
r"""
The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
"""
    x = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    w = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    x_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>
    w_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    dilations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    group = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    y = ResultDef(Attribute)  # should actually be outs TensorOf<[I32]>


@irdl_op_definition
class ONNXConvTransposeOp(Operation):
    name: str = "onnx.ConvTranspose"
    summary: str = \
r"""ONNX ConvTranspose operation"""
    description: str =\
r"""
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    W = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    dilations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    group = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    output_padding = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    output_shape = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXCosOp(Operation):
    name: str = "onnx.Cos"
    summary: str = \
r"""ONNX Cos operation"""
    description: str =\
r"""
Calculates the cosine of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXCoshOp(Operation):
    name: str = "onnx.Cosh"
    summary: str = \
r"""ONNX Cosh operation"""
    description: str =\
r"""
Calculates the hyperbolic cosine of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXCumSumOp(Operation):
    name: str = "onnx.CumSum"
    summary: str = \
r"""ONNX CumSum operation"""
    description: str =\
r"""
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.
"""
    x = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axis = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    exclusive = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    reverse = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXDepthToSpaceOp(Operation):
    name: str = "onnx.DepthToSpace"
    summary: str = \
r"""ONNX DepthToSpace operation"""
    description: str =\
r"""
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    blocksize = OptAttributeDef(Attribute)  # should actually be SI64Attr
    mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "DCR">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXDequantizeLinearOp(Operation):
    name: str = "onnx.DequantizeLinear"
    summary: str = \
r"""ONNX DequantizeLinear operation"""
    description: str =\
r"""
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
"""
    x = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>]>
    x_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    x_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, NoneType]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXDetOp(Operation):
    name: str = "onnx.Det"
    summary: str = \
r"""ONNX Det operation"""
    description: str =\
r"""
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `*, M, M`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `*`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: ``).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXDivOp(Operation):
    name: str = "onnx.Div"
    summary: str = \
r"""ONNX Div operation"""
    description: str =\
r"""
Performs element-wise binary division (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXDropoutOp(Operation):
    name: str = "onnx.Dropout"
    summary: str = \
r"""ONNX Dropout operation"""
    description: str =\
r"""
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    ratio = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    training_mode = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I1]>, NoneType]>
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    mask = ResultDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I1]>, NoneType]>


@irdl_op_definition
class ONNXDynamicQuantizeLinearOp(Operation):
    name: str = "onnx.DynamicQuantizeLinear"
    summary: str = \
r"""ONNX DynamicQuantizeLinear operation"""
    description: str =\
r"""
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (max(x) - min(x))/(qmax - qmin)
* where qmax and qmin are max and min values for quantization range .i.e 0, 255 in case of uint8
* data range is adjusted to include 0.
```
Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
* where qmax and qmin are max and min values for quantization range .i.e 0, 255 in case of uint8
* for saturation, it saturates to 0, 255 if it's uint8, or -127, 127 if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
```
Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
* for saturation, it saturates to 0, 255 if it's uint8, or -127, 127 if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
```
"""
    x = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    y = ResultDef(Attribute)  # should actually be outs TensorOf<[UI8]>
    y_scale = ResultDef(Attribute)  # should actually be TensorOf<[F32]>
    y_zero_point = ResultDef(Attribute)  # should actually be TensorOf<[UI8]>


@irdl_op_definition
class ONNXEinsumOp(Operation):
    name: str = "onnx.Einsum"
    summary: str = \
r"""ONNX Einsum operation"""
    description: str =\
r"""
An einsum of the form ```term1, term2 -> output-term``` produces an output tensor using the following equation
"""
    Inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>
    equation = OptAttributeDef(Attribute)  # should actually be StrAttr
    Output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXEluOp(Operation):
    name: str = "onnx.Elu"
    summary: str = \
r"""ONNX Elu operation"""
    description: str =\
r"""
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXEqualOp(Operation):
    name: str = "onnx.Equal"
    summary: str = \
r"""ONNX Equal operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXErfOp(Operation):
    name: str = "onnx.Erf"
    summary: str = \
r"""ONNX Erf operation"""
    description: str =\
r"""
Computes the error function of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXExpOp(Operation):
    name: str = "onnx.Exp"
    summary: str = \
r"""ONNX Exp operation"""
    description: str =\
r"""
Calculates the exponential of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXExpandOp(Operation):
    name: str = "onnx.Expand"
    summary: str = \
r"""ONNX Expand operation"""
    description: str =\
r"""
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    shape = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXEyeLikeOp(Operation):
    name: str = "onnx.EyeLike"
    summary: str = \
r"""ONNX EyeLike operation"""
    description: str =\
r"""
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>
    dtype = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    k = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>


@irdl_op_definition
class ONNXFlattenOp(Operation):
    name: str = "onnx.Flatten"
    summary: str = \
r"""ONNX Flatten operation"""
    description: str =\
r"""
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXFloorOp(Operation):
    name: str = "onnx.Floor"
    summary: str = \
r"""ONNX Floor operation"""
    description: str =\
r"""
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXGRUOp(Operation):
    name: str = "onnx.GRU"
    summary: str = \
r"""ONNX GRU operation"""
    description: str =\
r"""
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    W = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    R = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    sequence_lens = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I32]>, NoneType]>
    initial_h = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    activation_alpha = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    activation_beta = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    activations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    clip = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    direction = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "forward">
    hidden_size = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    layout = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    linear_before_reset = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    Y_h = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>


@irdl_op_definition
class ONNXGatherOp(Operation):
    name: str = "onnx.Gather"
    summary: str = \
r"""ONNX Gather operation"""
    description: str =\
r"""
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXGatherElementsOp(Operation):
    name: str = "onnx.GatherElements"
    summary: str = \
r"""ONNX GatherElements operation"""
    description: str =\
r"""
GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXGatherNDOp(Operation):
    name: str = "onnx.GatherND"
    summary: str = \
r"""ONNX GatherND operation"""
    description: str =\
r"""
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape-1 - 1 - b`.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    batch_dims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXGemmOp(Operation):
    name: str = "onnx.Gemm"
    summary: str = \
r"""ONNX Gemm operation"""
    description: str =\
r"""
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>
    C = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, NoneType]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    beta = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    transA = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    transB = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXGlobalAveragePoolOp(Operation):
    name: str = "onnx.GlobalAveragePool"
    summary: str = \
r"""ONNX GlobalAveragePool operation"""
    description: str =\
r"""
GlobalAveragePool consumes an input tensor X and applies average pooling across
the values in the same channel. This is equivalent to AveragePool with kernel size
equal to the spatial dimension of input tensor.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXGlobalLpPoolOp(Operation):
    name: str = "onnx.GlobalLpPool"
    summary: str = \
r"""ONNX GlobalLpPool operation"""
    description: str =\
r"""
GlobalLpPool consumes an input tensor X and applies lp pool pooling across
the values in the same channel. This is equivalent to LpPool with kernel size
equal to the spatial dimension of input tensor.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    p = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "2">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXGlobalMaxPoolOp(Operation):
    name: str = "onnx.GlobalMaxPool"
    summary: str = \
r"""ONNX GlobalMaxPool operation"""
    description: str =\
r"""
GlobalMaxPool consumes an input tensor X and applies max pooling across
the values in the same channel. This is equivalent to MaxPool with kernel size
equal to the spatial dimension of input tensor.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXGreaterOp(Operation):
    name: str = "onnx.Greater"
    summary: str = \
r"""ONNX Greater operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXGreaterOrEqualOp(Operation):
    name: str = "onnx.GreaterOrEqual"
    summary: str = \
r"""ONNX GreaterOrEqual operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXGridSampleOp(Operation):
    name: str = "onnx.GridSample"
    summary: str = \
r"""ONNX GridSample operation"""
    description: str =\
r"""
Given an `input` and a flow-field `grid`, computes the `output` using `input` values and pixel locations from `grid`.
Currently, only spatial (4-D) inputs are supported. For `input` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
the `output` will have shape (N, C, H_out, W_out).
For each output location `outputN, C, H_out, W_out`, the size-2 vector `gridN, H_out, W_out` specifies `input` pixel locations `x` and `y`,
which are used to interpolate the output value `outputN, C, H_out, W_out`.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    grid = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    align_corners = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "bilinear">
    padding_mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "zeros">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXHardSigmoidOp(Operation):
    name: str = "onnx.HardSigmoid"
    summary: str = \
r"""ONNX HardSigmoid operation"""
    description: str =\
r"""
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.2">
    beta = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.5">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXHardSwishOp(Operation):
    name: str = "onnx.HardSwish"
    summary: str = \
r"""ONNX HardSwish operation"""
    description: str =\
r"""
HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXHardmaxOp(Operation):
    name: str = "onnx.Hardmax"
    summary: str = \
r"""ONNX Hardmax operation"""
    description: str =\
r"""
The operator computes the hardmax values for the given input:
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXIdentityOp(Operation):
    name: str = "onnx.Identity"
    summary: str = \
r"""ONNX Identity operation"""
    description: str =\
r"""
Identity operator
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>


@irdl_op_definition
class ONNXIfOp(Operation):
    name: str = "onnx.If"
    summary: str = \
r"""ONNX If operation"""
    description: str =\
r"""
If conditional
"""
    cond = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>>


@irdl_op_definition
class ONNXInstanceNormalizationOp(Operation):
    name: str = "onnx.InstanceNormalization"
    summary: str = \
r"""ONNX InstanceNormalization operation"""
    description: str =\
r"""
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    scale = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    epsilon = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1e-05">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXIsInfOp(Operation):
    name: str = "onnx.IsInf"
    summary: str = \
r"""ONNX IsInf operation"""
    description: str =\
r"""
Map infinity to true and other values to false.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>
    detect_negative = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    detect_positive = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXIsNaNOp(Operation):
    name: str = "onnx.IsNaN"
    summary: str = \
r"""ONNX IsNaN operation"""
    description: str =\
r"""
Returns which elements of the input are NaN.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXLRNOp(Operation):
    name: str = "onnx.LRN"
    summary: str = \
r"""ONNX LRN operation"""
    description: str =\
r"""
Local Response Normalization proposed in the AlexNet paper(https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element Xn, c, d1, ..., dk in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
Xn, i, d1, ..., dk | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0001">
    beta = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.75">
    bias = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    size = OptAttributeDef(Attribute)  # should actually be SI64Attr
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXLSTMOp(Operation):
    name: str = "onnx.LSTM"
    summary: str = \
r"""ONNX LSTM operation"""
    description: str =\
r"""
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    W = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    R = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    sequence_lens = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I32]>, NoneType]>
    initial_h = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    initial_c = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    P = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    activation_alpha = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    activation_beta = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    activations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    clip = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    direction = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "forward">
    hidden_size = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    input_forget = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    layout = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    Y_h = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    Y_c = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>


@irdl_op_definition
class ONNXLeakyReluOp(Operation):
    name: str = "onnx.LeakyRelu"
    summary: str = \
r"""ONNX LeakyRelu operation"""
    description: str =\
r"""
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.01">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXLessOp(Operation):
    name: str = "onnx.Less"
    summary: str = \
r"""ONNX Less operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXLessOrEqualOp(Operation):
    name: str = "onnx.LessOrEqual"
    summary: str = \
r"""ONNX LessOrEqual operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXLogOp(Operation):
    name: str = "onnx.Log"
    summary: str = \
r"""ONNX Log operation"""
    description: str =\
r"""
Calculates the natural log of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXLogSoftmaxOp(Operation):
    name: str = "onnx.LogSoftmax"
    summary: str = \
r"""ONNX LogSoftmax operation"""
    description: str =\
r"""
The operator computes the log of softmax values for the given input:
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXLoopOp(Operation):
    name: str = "onnx.Loop"
    summary: str = \
r"""ONNX Loop operation"""
    description: str =\
r"""
Generic Looping construct. This loop has multiple termination conditions:
"""
    M = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    cond = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I1]>, NoneType]>
    v_initial = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>>
    v_final_and_scan_outputs = VarResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>>


@irdl_op_definition
class ONNXLpNormalizationOp(Operation):
    name: str = "onnx.LpNormalization"
    summary: str = \
r"""ONNX LpNormalization operation"""
    description: str =\
r"""
Given a matrix, apply Lp-normalization along the provided axis.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    p = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "2">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXLpPoolOp(Operation):
    name: str = "onnx.LpPool"
    summary: str = \
r"""ONNX LpPool operation"""
    description: str =\
r"""
LpPool consumes an input tensor X and applies Lp pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
Lp pooling consisting of computing the Lp norm on all values of a subset
of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    p = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "2">
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXMatMulOp(Operation):
    name: str = "onnx.MatMul"
    summary: str = \
r"""ONNX MatMul operation"""
    description: str =\
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXMatMulIntegerOp(Operation):
    name: str = "onnx.MatMulInteger"
    summary: str = \
r"""ONNX MatMulInteger operation"""
    description: str =\
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    a_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>
    b_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[I32]>


@irdl_op_definition
class ONNXMaxOp(Operation):
    name: str = "onnx.Max"
    summary: str = \
r"""ONNX Max operation"""
    description: str =\
r"""
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
"""
    data_0 = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>
    max = VarResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXMaxPoolOp(Operation):
    name: str = "onnx.MaxPool"
    summary: str = \
r"""ONNX MaxPool operation"""
    description: str =\
r"""
MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
```
output_spatial_shapei = floor((input_spatial_shapei + pad_shapei - ((kernel_spatial_shapei - 1) * dilationsi + 1)) / strides_spatial_shapei + 1)
```
or
```
output_spatial_shapei = ceil((input_spatial_shapei + pad_shapei - ((kernel_spatial_shapei - 1) * dilationsi + 1)) / strides_spatial_shapei + 1)
```
if ceil_mode is enabled
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    ceil_mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    dilations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    storage_order = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>
    Indices = ResultDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>


@irdl_op_definition
class ONNXMaxRoiPoolOp(Operation):
    name: str = "onnx.MaxRoiPool"
    summary: str = \
r"""ONNX MaxRoiPool operation"""
    description: str =\
r"""
ROI max pool consumes an input tensor X and region of interests (RoIs) to
apply max pooling across each RoI, to produce output 4-D tensor of shape
(num_rois, channels, pooled_shape0, pooled_shape1).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    rois = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    pooled_shape = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    spatial_scale = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXMaxUnpoolOp(Operation):
    name: str = "onnx.MaxUnpool"
    summary: str = \
r"""ONNX MaxUnpool operation"""
    description: str =\
r"""
MaxUnpool essentially computes the partial inverse of the MaxPool op.
The input information to this op is typically the the output information from a MaxPool op. The first
input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
The third (optional) input is a tensor that specifies the output size of the unpooling operation.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    I = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    output_shape = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXMeanOp(Operation):
    name: str = "onnx.Mean"
    summary: str = \
r"""ONNX Mean operation"""
    description: str =\
r"""
Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
"""
    data_0 = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>
    mean = VarResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXMeanVarianceNormalizationOp(Operation):
    name: str = "onnx.MeanVarianceNormalization"
    summary: str = \
r"""ONNX MeanVarianceNormalization operation"""
    description: str =\
r"""
A MeanVarianceNormalization Function: Perform mean variance normalization
on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedAttr<I64ArrayAttr, "{0, 2, 3}">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXMinOp(Operation):
    name: str = "onnx.Min"
    summary: str = \
r"""ONNX Min operation"""
    description: str =\
r"""
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
"""
    data_0 = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>
    min = VarResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXModOp(Operation):
    name: str = "onnx.Mod"
    summary: str = \
r"""ONNX Mod operation"""
    description: str =\
r"""
Performs element-wise binary modulus (with Numpy-style broadcasting support).
The sign of the remainder is the same as that of the Divisor.
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    fmod = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    C = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXMulOp(Operation):
    name: str = "onnx.Mul"
    summary: str = \
r"""ONNX Mul operation"""
    description: str =\
r"""
Performs element-wise binary multiplication (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXMultinomialOp(Operation):
    name: str = "onnx.Multinomial"
    summary: str = \
r"""ONNX Multinomial operation"""
    description: str =\
r"""
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    dtype = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "6">
    sample_size = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>


@irdl_op_definition
class ONNXNegOp(Operation):
    name: str = "onnx.Neg"
    summary: str = \
r"""ONNX Neg operation"""
    description: str =\
r"""
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXNegativeLogLikelihoodLossOp(Operation):
    name: str = "onnx.NegativeLogLikelihoodLoss"
    summary: str = \
r"""ONNX NegativeLogLikelihoodLoss operation"""
    description: str =\
r"""
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its \input\ tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The \input\ tensor contains log-probabilities for inputn, :, d_1, d_2,..., d_k being in a class of 0, C).
The operator's \target\ input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for inputn, :, d_1, d_2,...d_k being classified as class c = targetnd_1d_2...d_k is computed as:
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    target = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    weight = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    ignore_index = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    reduction = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "mean">
    loss = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXNonMaxSuppressionOp(Operation):
    name: str = "onnx.NonMaxSuppression"
    summary: str = \
r"""ONNX NonMaxSuppression operation"""
    description: str =\
r"""
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
"""
    boxes = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    scores = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    max_output_boxes_per_class = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    iou_threshold = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[F32]>, NoneType]>
    score_threshold = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[F32]>, NoneType]>
    center_point_box = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    selected_indices = ResultDef(
        Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXNonZeroOp(Operation):
    name: str = "onnx.NonZero"
    summary: str = \
r"""ONNX NonZero operation"""
    description: str =\
r"""
Returns the indices of the elements that are non-zero
(in row-major order - by dimension).
NonZero behaves similar to numpy.nonzero:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXNotOp(Operation):
    name: str = "onnx.Not"
    summary: str = \
r"""ONNX Not operation"""
    description: str =\
r"""
Returns the negation of the input tensor element-wise.
"""
    X = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXOneHotOp(Operation):
    name: str = "onnx.OneHot"
    summary: str = \
r"""ONNX OneHot operation"""
    description: str =\
r"""
Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
off_value, on_value. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range -depth, depth-1 will result in one-hot representation with all 'off_value' values in the
output tensor.
"""
    indices = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    depth = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    values = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXOptionalOp(Operation):
    name: str = "onnx.Optional"
    summary: str = \
r"""ONNX Optional operation"""
    description: str =\
r"""
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, NoneType]>
    type = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<TypeAttr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>


@irdl_op_definition
class ONNXOptionalGetElementOp(Operation):
    name: str = "onnx.OptionalGetElement"
    summary: str = \
r"""ONNX OptionalGetElement operation"""
    description: str =\
r"""
Outputs the element in the optional-type input. It is an error if the input value does not have an element
and the behavior is undefined in this case.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>


@irdl_op_definition
class ONNXOptionalHasElementOp(Operation):
    name: str = "onnx.OptionalHasElement"
    summary: str = \
r"""ONNX OptionalHasElement operation"""
    description: str =\
r"""
Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>
    output = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXOrOp(Operation):
    name: str = "onnx.Or"
    summary: str = \
r"""ONNX Or operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    B = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXPReluOp(Operation):
    name: str = "onnx.PRelu"
    summary: str = \
r"""ONNX PRelu operation"""
    description: str =\
r"""
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>
    slope = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>


@irdl_op_definition
class ONNXPadOp(Operation):
    name: str = "onnx.Pad"
    summary: str = \
r"""ONNX Pad operation"""
    description: str =\
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    pads = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    constant_value = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "constant">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXPadV11Op(Operation):
    name: str = "onnx.PadV11"
    summary: str = \
r"""ONNX Pad operation"""
    description: str =\
r"""
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    pads = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    constant_value = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "constant">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXPadV2Op(Operation):
    name: str = "onnx.PadV2"
    summary: str = \
r"""ONNX Pad operation"""
    description: str =\
r"""
Given `data` tensor, pads, mode, and value.
Example:
Insert 0 pads to the beginning of the second dimension.
data = 
1.0, 1.2,
2.3, 3.4,
4.5, 5.7,

pads = 0, 2, 0, 0
output = 

0.0, 0.0, 1.0, 1.2,
0.0, 0.0, 2.3, 3.4,
0.0, 0.0, 4.5, 5.7,
,

"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "constant">
    pads = OptAttributeDef(Attribute)  # should actually be I64ArrayAttr
    value = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXPowOp(Operation):
    name: str = "onnx.Pow"
    summary: str = \
r"""ONNX Pow operation"""
    description: str =\
r"""
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Z = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXQLinearConvOp(Operation):
    name: str = "onnx.QLinearConv"
    summary: str = \
r"""ONNX QLinearConv operation"""
    description: str =\
r"""
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
"""
    x = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    x_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    x_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    w = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    w_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    w_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    y_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    y_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    B = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I32]>, NoneType]>
    auto_pad = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "NOTSET">
    dilations = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    group = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    kernel_shape = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    pads = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    strides = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>


@irdl_op_definition
class ONNXQLinearMatMulOp(Operation):
    name: str = "onnx.QLinearMatMul"
    summary: str = \
r"""ONNX QLinearMatMul operation"""
    description: str =\
r"""
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape M, K then zero point and scale tensor may be
an M element vector v_1, v_2, ..., v_M for per row quantization and K element vector of shape v_1, v_2, ..., v_K
for per column quantization. If the input is N-D tensor with shape D1, D2, M, K then zero point and scale tensor may
have shape D1, D2, M, 1 for per row quantization and shape D1, D2, 1, K for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
"""
    a = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    a_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    a_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    b = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    b_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    b_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    y_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    y_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>
    y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>


@irdl_op_definition
class ONNXQuantizeLinearOp(Operation):
    name: str = "onnx.QuantizeLinear"
    summary: str = \
r"""ONNX QuantizeLinear operation"""
    description: str =\
r"""
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to 0, 255 if it's uint8, or -128, 127 if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
"""
    x = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>]>
    y_scale = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    y_zero_point = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>


@irdl_op_definition
class ONNXRNNOp(Operation):
    name: str = "onnx.RNN"
    summary: str = \
r"""ONNX RNN operation"""
    description: str =\
r"""
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    W = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    R = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    sequence_lens = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I32]>, NoneType]>
    initial_h = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    activation_alpha = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    activation_beta = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    activations = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedAttr<StrArrayAttr, "{\"Tanh\", \"Tanh\"}">
    clip = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    direction = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "forward">
    hidden_size = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    layout = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    Y_h = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>


@irdl_op_definition
class ONNXRandomNormalOp(Operation):
    name: str = "onnx.RandomNormal"
    summary: str = \
r"""ONNX RandomNormal operation"""
    description: str =\
r"""
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.
"""
    dtype = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    mean = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    scale = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    shape = OptAttributeDef(Attribute)  # should actually be I64ArrayAttr
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXRandomNormalLikeOp(Operation):
    name: str = "onnx.RandomNormalLike"
    summary: str = \
r"""ONNX RandomNormalLike operation"""
    description: str =\
r"""
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    dtype = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    mean = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    scale = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXRandomUniformOp(Operation):
    name: str = "onnx.RandomUniform"
    summary: str = \
r"""ONNX RandomUniform operation"""
    description: str =\
r"""
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.
"""
    dtype = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    high = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    low = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    shape = OptAttributeDef(Attribute)  # should actually be I64ArrayAttr
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXRandomUniformLikeOp(Operation):
    name: str = "onnx.RandomUniformLike"
    summary: str = \
r"""ONNX RandomUniformLike operation"""
    description: str =\
r"""
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    dtype = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    high = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    low = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    seed = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXRangeOp(Operation):
    name: str = "onnx.Range"
    summary: str = \
r"""ONNX Range operation"""
    description: str =\
r"""
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).
"""
    start = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>
    limit = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>
    delta = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>


@irdl_op_definition
class ONNXReciprocalOp(Operation):
    name: str = "onnx.Reciprocal"
    summary: str = \
r"""ONNX Reciprocal operation"""
    description: str =\
r"""
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceL1Op(Operation):
    name: str = "onnx.ReduceL1"
    summary: str = \
r"""ONNX ReduceL1 operation"""
    description: str =\
r"""
Computes the L1 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceL2Op(Operation):
    name: str = "onnx.ReduceL2"
    summary: str = \
r"""ONNX ReduceL2 operation"""
    description: str =\
r"""
Computes the L2 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceLogSumOp(Operation):
    name: str = "onnx.ReduceLogSum"
    summary: str = \
r"""ONNX ReduceLogSum operation"""
    description: str =\
r"""
Computes the log sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceLogSumExpOp(Operation):
    name: str = "onnx.ReduceLogSumExp"
    summary: str = \
r"""ONNX ReduceLogSumExp operation"""
    description: str =\
r"""
Computes the log sum exponent of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceMaxOp(Operation):
    name: str = "onnx.ReduceMax"
    summary: str = \
r"""ONNX ReduceMax operation"""
    description: str =\
r"""
Computes the max of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>


@irdl_op_definition
class ONNXReduceMeanOp(Operation):
    name: str = "onnx.ReduceMean"
    summary: str = \
r"""ONNX ReduceMean operation"""
    description: str =\
r"""
Computes the mean of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceMinOp(Operation):
    name: str = "onnx.ReduceMin"
    summary: str = \
r"""ONNX ReduceMin operation"""
    description: str =\
r"""
Computes the min of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>


@irdl_op_definition
class ONNXReduceProdOp(Operation):
    name: str = "onnx.ReduceProd"
    summary: str = \
r"""ONNX ReduceProd operation"""
    description: str =\
r"""
Computes the product of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceSumOp(Operation):
    name: str = "onnx.ReduceSum"
    summary: str = \
r"""ONNX ReduceSum operation"""
    description: str =\
r"""
Computes the sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    noop_with_empty_axes = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReduceSumV11Op(Operation):
    name: str = "onnx.ReduceSumV11"
    summary: str = \
r"""ONNX ReduceSum operation"""
    description: str =\
r"""
Computes the sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXReduceSumSquareOp(Operation):
    name: str = "onnx.ReduceSumSquare"
    summary: str = \
r"""ONNX ReduceSumSquare operation"""
    description: str =\
r"""
Computes the sum square of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    reduced = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReluOp(Operation):
    name: str = "onnx.Relu"
    summary: str = \
r"""ONNX Relu operation"""
    description: str =\
r"""
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXReshapeOp(Operation):
    name: str = "onnx.Reshape"
    summary: str = \
r"""ONNX Reshape operation"""
    description: str =\
r"""
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    shape = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    allowzero = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    reshaped = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXResizeOp(Operation):
    name: str = "onnx.Resize"
    summary: str = \
r"""ONNX Resize operation"""
    description: str =\
r"""
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \\sizes\\ is not specified.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    roi = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    scales = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[F32]>, NoneType]>
    sizes = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    coordinate_transformation_mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "half_pixel">
    cubic_coeff_a = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "-0.75">
    exclude_outside = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    extrapolation_value = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "nearest">
    nearest_mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "round_prefer_floor">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXResizeV11Op(Operation):
    name: str = "onnx.ResizeV11"
    summary: str = \
r"""ONNX Resize operation"""
    description: str =\
r"""
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \\sizes\\ is not specified.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    roi = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    scales = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    sizes = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    coordinate_transformation_mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "half_pixel">
    cubic_coeff_a = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "-0.75">
    exclude_outside = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    extrapolation_value = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "nearest">
    nearest_mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "round_prefer_floor">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXResizeV10Op(Operation):
    name: str = "onnx.ResizeV10"
    summary: str = \
r"""ONNX Resize operation"""
    description: str =\
r"""
Resize the input tensor.
Each dimension value of the output tensor is:
output_dimension = floor(input_dimension * scale).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    scales = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "nearest">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXReverseSequenceOp(Operation):
    name: str = "onnx.ReverseSequence"
    summary: str = \
r"""ONNX ReverseSequence operation"""
    description: str =\
r"""
Reverse batch of sequences having different lengths specified by `sequence_lens`.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    sequence_lens = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    batch_axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    time_axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXRoiAlignOp(Operation):
    name: str = "onnx.RoiAlign"
    summary: str = \
r"""ONNX RoiAlign operation"""
    description: str =\
r"""
Region of Interest (RoI) align operation described in the
Mask R-CNN paper(https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    rois = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    batch_indices = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    coordinate_transformation_mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "half_pixel">
    mode = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "avg">
    output_height = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    output_width = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    sampling_ratio = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    spatial_scale = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXRoundOp(Operation):
    name: str = "onnx.Round"
    summary: str = \
r"""ONNX Round operation"""
    description: str =\
r"""
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halfs, the rule is to round them to the nearest even integer.
The output tensor has the same shape and type as the input.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXScanOp(Operation):
    name: str = "onnx.Scan"
    summary: str = \
r"""ONNX Scan operation"""
    description: str =\
r"""
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.
"""
    initial_state_and_scan_inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>
    num_scan_inputs = OptAttributeDef(Attribute)  # should actually be SI64Attr
    scan_input_axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    scan_input_directions = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    scan_output_axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    scan_output_directions = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    final_state_and_scan_outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>


@irdl_op_definition
class ONNXScatterOp(Operation):
    name: str = "onnx.Scatter"
    summary: str = \
r"""ONNX Scatter operation"""
    description: str =\
r"""
This operator is deprecated. Please use ScatterElements, which provides the same functionality.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    updates = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXScatterElementsOp(Operation):
    name: str = "onnx.ScatterElements"
    summary: str = \
r"""ONNX ScatterElements operation"""
    description: str =\
r"""
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    updates = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    reduction = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "none">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXScatterNDOp(Operation):
    name: str = "onnx.ScatterND"
    summary: str = \
r"""ONNX ScatterND operation"""
    description: str =\
r"""
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape-1 - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`. Note that `indices` should not have duplicate entries.
That is, two or more `updates` for the same index-location is not supported.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    updates = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    reduction = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "none">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXSeluOp(Operation):
    name: str = "onnx.Selu"
    summary: str = \
r"""ONNX Selu operation"""
    description: str =\
r"""
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.67326">
    gamma = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0507">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXSequenceAtOp(Operation):
    name: str = "onnx.SequenceAt"
    summary: str = \
r"""ONNX SequenceAt operation"""
    description: str =\
r"""
Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `-n, n - 1`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
"""
    input_sequence = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>
    position = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    tensor = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXSequenceConstructOp(Operation):
    name: str = "onnx.SequenceConstruct"
    summary: str = \
r"""ONNX SequenceConstruct operation"""
    description: str =\
r"""
Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
"""
    inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>
    output_sequence = VarResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>


@irdl_op_definition
class ONNXSequenceEmptyOp(Operation):
    name: str = "onnx.SequenceEmpty"
    summary: str = \
r"""ONNX SequenceEmpty operation"""
    description: str =\
r"""
Construct an empty tensor sequence, with given data type.
"""
    dtype = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>


@irdl_op_definition
class ONNXSequenceEraseOp(Operation):
    name: str = "onnx.SequenceErase"
    summary: str = \
r"""ONNX SequenceErase operation"""
    description: str =\
r"""
Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `-n, n - 1`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
"""
    input_sequence = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>
    position = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>
    output_sequence = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>


@irdl_op_definition
class ONNXSequenceInsertOp(Operation):
    name: str = "onnx.SequenceInsert"
    summary: str = \
r"""ONNX SequenceInsert operation"""
    description: str =\
r"""
Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `-n, n`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
"""
    input_sequence = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>
    tensor = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    position = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>
    output_sequence = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>


@irdl_op_definition
class ONNXSequenceLengthOp(Operation):
    name: str = "onnx.SequenceLength"
    summary: str = \
r"""ONNX SequenceLength operation"""
    description: str =\
r"""
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
"""
    input_sequence = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>
    length = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXShapeOp(Operation):
    name: str = "onnx.Shape"
    summary: str = \
r"""ONNX Shape operation"""
    description: str =\
r"""
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clipped to the range 0, r-1, where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    end = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    start = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    shape = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXShrinkOp(Operation):
    name: str = "onnx.Shrink"
    summary: str = \
r"""ONNX Shrink operation"""
    description: str =\
r"""
Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    bias = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    lambd = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.5">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXSigmoidOp(Operation):
    name: str = "onnx.Sigmoid"
    summary: str = \
r"""ONNX Sigmoid operation"""
    description: str =\
r"""
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXSignOp(Operation):
    name: str = "onnx.Sign"
    summary: str = \
r"""ONNX Sign operation"""
    description: str =\
r"""
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXSinOp(Operation):
    name: str = "onnx.Sin"
    summary: str = \
r"""ONNX Sin operation"""
    description: str =\
r"""
Calculates the sine of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXSinhOp(Operation):
    name: str = "onnx.Sinh"
    summary: str = \
r"""ONNX Sinh operation"""
    description: str =\
r"""
Calculates the hyperbolic sine of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXSizeOp(Operation):
    name: str = "onnx.Size"
    summary: str = \
r"""ONNX Size operation"""
    description: str =\
r"""
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    size = ResultDef(Attribute)  # should actually be outs TensorOf<[I64]>


@irdl_op_definition
class ONNXSliceOp(Operation):
    name: str = "onnx.Slice"
    summary: str = \
r"""ONNX Slice operation"""
    description: str =\
r"""
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represents number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`
when sclicing forward and 'INT_MIN' when slicing backward.
If a negative value is passed for step, it represents slicing backward.
However step value cannot be 0.
If `axes` are omitted, they are set to `0, ..., ndim-1`.
If `steps` are omitted, they are set to `1, ..., 1` of length `len(starts)`
Example 1:
data = 
1, 2, 3, 4,
5, 6, 7, 8,

axes = 0, 1
starts = 1, 0
ends = 2, 3
steps = 1, 2
result = 
5, 7,

Example 2:
data = 
1, 2, 3, 4,
5, 6, 7, 8,

starts = 0, 1
ends = -1, 1000
result = 
2, 3, 4,

"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    starts = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    ends = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    axes = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>
    steps = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXSoftmaxOp(Operation):
    name: str = "onnx.Softmax"
    summary: str = \
r"""ONNX Softmax operation"""
    description: str =\
r"""
The operator computes the normalized exponential values for the given input:
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXSoftmaxCrossEntropyLossOp(Operation):
    name: str = "onnx.SoftmaxCrossEntropyLoss"
    summary: str = \
r"""ONNX SoftmaxCrossEntropyLoss operation"""
    description: str =\
r"""
Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and Li,j_1j_2...j_k denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.
"""
    scores = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    labels = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>
    weights = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>
    ignore_index = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    reduction = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "mean">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    log_prob = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>


@irdl_op_definition
class ONNXSoftplusOp(Operation):
    name: str = "onnx.Softplus"
    summary: str = \
r"""ONNX Softplus operation"""
    description: str =\
r"""
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXSoftsignOp(Operation):
    name: str = "onnx.Softsign"
    summary: str = \
r"""ONNX Softsign operation"""
    description: str =\
r"""
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXSpaceToDepthOp(Operation):
    name: str = "onnx.SpaceToDepth"
    summary: str = \
r"""ONNX SpaceToDepth operation"""
    description: str =\
r"""
SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    blocksize = OptAttributeDef(Attribute)  # should actually be SI64Attr
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXSplitOp(Operation):
    name: str = "onnx.Split"
    summary: str = \
r"""ONNX Split operation"""
    description: str =\
r"""
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    split = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>


@irdl_op_definition
class ONNXSplitV11Op(Operation):
    name: str = "onnx.SplitV11"
    summary: str = \
r"""ONNX Split operation"""
    description: str =\
r"""
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    split = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>


@irdl_op_definition
class ONNXSplitToSequenceOp(Operation):
    name: str = "onnx.SplitToSequence"
    summary: str = \
r"""ONNX SplitToSequence operation"""
    description: str =\
r"""
Split a tensor into a sequence of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into equally sized chunks(if possible).
Last chunk will be smaller if the 'input' size along the given axis 'axis' is not divisible
by 'split'.
Otherwise, the tensor is split into 'size(split)' chunks, with lengths of the parts on 'axis'
specified in 'split'. In this scenario, the sum of entries in 'split' must be equal to the
dimension size of input tensor on 'axis'.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    split = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    keepdims = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    output_sequence = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>


@irdl_op_definition
class ONNXSqrtOp(Operation):
    name: str = "onnx.Sqrt"
    summary: str = \
r"""ONNX Sqrt operation"""
    description: str =\
r"""
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXSqueezeOp(Operation):
    name: str = "onnx.Squeeze"
    summary: str = \
r"""ONNX Squeeze operation"""
    description: str =\
r"""
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axes = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    squeezed = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXSqueezeV11Op(Operation):
    name: str = "onnx.SqueezeV11"
    summary: str = \
r"""ONNX Squeeze operation"""
    description: str =\
r"""
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    squeezed = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXStringNormalizerOp(Operation):
    name: str = "onnx.StringNormalizer"
    summary: str = \
r"""ONNX StringNormalizer operation"""
    description: str =\
r"""
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in \stopwords\ attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the \case_change_action\ attribute.
This operator only accepts C- and 1, C-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape 1
if input shape is C and shape 1, 1 if input shape is 1, C.
"""
    X = OperandDef(Attribute)  # should actually be TensorOf<[StringType]>
    case_change_action = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    is_case_sensitive = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    locale = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrAttr>
    stopwords = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[StringType]>


@irdl_op_definition
class ONNXSubOp(Operation):
    name: str = "onnx.Sub"
    summary: str = \
r"""ONNX Sub operation"""
    description: str =\
r"""
Performs element-wise binary subtraction (with Numpy-style broadcasting support).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXSumOp(Operation):
    name: str = "onnx.Sum"
    summary: str = \
r"""ONNX Sum operation"""
    description: str =\
r"""
Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
"""
    data_0 = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>
    sum = VarResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXTanOp(Operation):
    name: str = "onnx.Tan"
    summary: str = \
r"""ONNX Tan operation"""
    description: str =\
r"""
Calculates the tangent of the given input tensor, element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXTanhOp(Operation):
    name: str = "onnx.Tanh"
    summary: str = \
r"""ONNX Tanh operation"""
    description: str =\
r"""
Calculates the hyperbolic tangent of the given input tensor element-wise.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>


@irdl_op_definition
class ONNXTfIdfVectorizerOp(Operation):
    name: str = "onnx.TfIdfVectorizer"
    summary: str = \
r"""ONNX TfIdfVectorizer operation"""
    description: str =\
r"""
This transform extracts n-grams from the input sequence and save them as a vector. Input can
be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
More specifically, if input shape is C, the corresponding output shape would be max(ngram_indexes) + 1.
If input shape is N, C, this operator produces a N, max(ngram_indexes) + 1-tensor.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I32]>, TensorOf<[I64]>]>
    max_gram_length = OptAttributeDef(Attribute)  # should actually be SI64Attr
    max_skip_count = OptAttributeDef(Attribute)  # should actually be SI64Attr
    min_gram_length = OptAttributeDef(Attribute)  # should actually be SI64Attr
    mode = OptAttributeDef(Attribute)  # should actually be StrAttr
    ngram_counts = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    ngram_indexes = OptAttributeDef(
        Attribute)  # should actually be I64ArrayAttr
    pool_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    pool_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    weights = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXThresholdedReluOp(Operation):
    name: str = "onnx.ThresholdedRelu"
    summary: str = \
r"""ONNX ThresholdedRelu operation"""
    description: str =\
r"""
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1.0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>


@irdl_op_definition
class ONNXTileOp(Operation):
    name: str = "onnx.Tile"
    summary: str = \
r"""ONNX Tile operation"""
    description: str =\
r"""
Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = 1, 2, 3, 4, B = 1, 2, tile(A, B) = 1, 2, 1, 2, 3, 4, 3, 4
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    repeats = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXTopKOp(Operation):
    name: str = "onnx.TopK"
    summary: str = \
r"""ONNX TopK operation"""
    description: str =\
r"""
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape a_1, a_2, ..., a_n, r and integer argument k, return two outputs:
-Value tensor of shape a_1, a_2, ..., a_axis-1, k, a_axis+1, ... a_n
which contains the values of the top k elements along the specified axis
-Index tensor of shape a_1, a_2, ..., a_axis-1, k, a_axis+1, ... a_n which
contains the indices of the top k elements (original indices from the input
tensor).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    K = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    axis = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    largest = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    sorted = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    Values = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    Indices = ResultDef(Attribute)  # should actually be TensorOf<[I64]>


@irdl_op_definition
class ONNXTransposeOp(Operation):
    name: str = "onnx.Transpose"
    summary: str = \
r"""ONNX Transpose operation"""
    description: str =\
r"""
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    perm = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    transposed = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXTriluOp(Operation):
    name: str = "onnx.Trilu"
    summary: str = \
r"""ONNX Trilu operation"""
    description: str =\
r"""
Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute \upper\ determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the \upper\ attribute is true.
Trilu takes one input tensor of shape *, N, M, where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    k = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    upper = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXUniqueOp(Operation):
    name: str = "onnx.Unique"
    summary: str = \
r"""ONNX Unique operation"""
    description: str =\
r"""
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axis = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    sorted = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    indices = ResultDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    inverse_indices = ResultDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    counts = ResultDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>


@irdl_op_definition
class ONNXUnsqueezeOp(Operation):
    name: str = "onnx.Unsqueeze"
    summary: str = \
r"""ONNX Unsqueeze operation"""
    description: str =\
r"""
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axes = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    expanded = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXUnsqueezeV11Op(Operation):
    name: str = "onnx.UnsqueezeV11"
    summary: str = \
r"""ONNX Unsqueeze operation"""
    description: str =\
r"""
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required argument `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    axes = OptAttributeDef(Attribute)  # should actually be I64ArrayAttr
    expanded = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXUpsampleOp(Operation):
    name: str = "onnx.Upsample"
    summary: str = \
r"""ONNX Upsample operation"""
    description: str =\
r"""
Upsample the input tensor.
Each dimension value of the output tensor is:
output_dimension = floor(input_dimension * scale).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    scales = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "nearest">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXUpsampleV9Op(Operation):
    name: str = "onnx.UpsampleV9"
    summary: str = \
r"""ONNX Upsample operation"""
    description: str =\
r"""
Upsample the input tensor.
Each dimension value of the output tensor is:
output_dimension = floor(input_dimension * scale).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    scales = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "nearest">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXUpsampleV7Op(Operation):
    name: str = "onnx.UpsampleV7"
    summary: str = \
r"""ONNX Upsample operation"""
    description: str =\
r"""
Upsample the input tensor.
Each dimension value of the output tensor is:
output_dimension = floor(input_dimension * scale).
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    mode = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "nearest">
    scales = OptAttributeDef(Attribute)  # should actually be F32ArrayAttr
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXWhereOp(Operation):
    name: str = "onnx.Where"
    summary: str = \
r"""ONNX Where operation"""
    description: str =\
r"""
Return elements, either from X or Y, depending on condition.
Where behaves like
numpy.where(https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.
"""
    condition = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    Y = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


@irdl_op_definition
class ONNXXorOp(Operation):
    name: str = "onnx.Xor"
    summary: str = \
r"""ONNX Xor operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
"""
    A = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    B = OperandDef(Attribute)  # should actually be TensorOf<[I1]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


@irdl_op_definition
class ONNXArrayFeatureExtractorOp(Operation):
    name: str = "onnx.ArrayFeatureExtractor"
    summary: str = \
r"""ONNX ArrayFeatureExtractor operation"""
    description: str =\
r"""
Select elements of the input tensor based on the indices passed.<br>
The indices are applied to the last axes of the tensor.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>, TensorOf<[StringType]>]>
    Y = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    Z = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>, TensorOf<[StringType]>]>


@irdl_op_definition
class ONNXBinarizerOp(Operation):
    name: str = "onnx.Binarizer"
    summary: str = \
r"""ONNX Binarizer operation"""
    description: str =\
r"""
Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    threshold = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>


@irdl_op_definition
class ONNXCastMapOp(Operation):
    name: str = "onnx.CastMap"
    summary: str = \
r"""ONNX CastMap operation"""
    description: str =\
r"""
Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
If using sparse packing, the key cannot exceed the max_map-1 value.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TupleOf<[I64, StringType]>, TupleOf<[I64, F32]>]>
    cast_to = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "TO_FLOAT">
    map_form = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "DENSE">
    max_map = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[F32]>, TensorOf<[I64]>]>


@irdl_op_definition
class ONNXCategoryMapperOp(Operation):
    name: str = "onnx.CategoryMapper"
    summary: str = \
r"""ONNX CategoryMapper operation"""
    description: str =\
r"""
Converts strings to integers and vice versa.<br>
Two sequences of equal length are used to map between integers and strings,
with strings and integers at the same index detailing the mapping.<br>
Each operator converts either integers to strings or strings to integers, depending
on which default value attribute is provided. Only one default value attribute
should be defined.<br>
If the string default value is set, it will convert integers to strings.
If the int default value is set, it will convert strings to integers.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>]>
    cats_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    cats_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    default_int64 = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    default_string = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "_Unused">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>]>


@irdl_op_definition
class ONNXDictVectorizerOp(Operation):
    name: str = "onnx.DictVectorizer"
    summary: str = \
r"""ONNX DictVectorizer operation"""
    description: str =\
r"""
Uses an index mapping to convert a dictionary to an array.<br>
Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
the key type. The index into the vocabulary array at which the key is found is then
used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
The key type of the input map must correspond to the element type of the defined vocabulary attribute.
Therefore, the output array will be equal in length to the index mapping vector parameter.
All keys in the input dictionary must be present in the index mapping vector.
For each item in the input dictionary, insert its value in the output array.
Any keys not present in the input dictionary, will be zero in the output array.<br>
For example: if the ``string_vocabulary`` parameter is set to ``\a\, \c\, \b\, \z\``,
then an input of ``\a\: 4, \c\: 8`` will produce an output of ``4, 8, 0, 0``.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TupleOf<[StringType, I64]>, TupleOf<[I64, StringType]>, TupleOf<[I64, F32]>, TupleOf<[I64, F64]>, TupleOf<[StringType, F32]>, TupleOf<[StringType, F64]>]>
    int64_vocabulary = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    string_vocabulary = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[I64]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>]>


@irdl_op_definition
class ONNXFeatureVectorizerOp(Operation):
    name: str = "onnx.FeatureVectorizer"
    summary: str = \
r"""ONNX FeatureVectorizer operation"""
    description: str =\
r"""
Concatenates input tensors into one continuous output.<br>
All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as 1,C.
Inputs are copied to the output maintaining the order of the input arguments.<br>
All inputs must be integers or floats, while the output will be all floating point values.
"""
    X = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F32]>, TensorOf<[F64]>]>>
    inputdimensions = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXImputerOp(Operation):
    name: str = "onnx.Imputer"
    summary: str = \
r"""ONNX Imputer operation"""
    description: str =\
r"""
Replaces inputs that equal one value with another, leaving all other elements alone.<br>
This operator is typically used to replace missing values in situations where they have a canonical
representation, such as -1, 0, NaN, or some extreme value.<br>
One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
which one depends on whether floats or integers are being processed.<br>
The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape *,F, then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    imputed_value_floats = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    imputed_value_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    replaced_value_float = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    replaced_value_int64 = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>


@irdl_op_definition
class ONNXLabelEncoderOp(Operation):
    name: str = "onnx.LabelEncoder"
    summary: str = \
r"""ONNX LabelEncoder operation"""
    description: str =\
r"""
Maps each element in the input tensor to another value.<br>
The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value.<br>
Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is \Amy\, \Sally\, 'values_int64s' is 5, 6,
and 'default_int64' is '-1'.  The input \Dori\, \Amy\, \Amy\, \Sally\,
\Sally\ would be mapped to -1, 5, 5, 6, 6.<br>
Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
For key look-up, bit-wise comparison is used so even a float NaN can be
mapped to a value in 'values_*' attribute.<br>
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>, TensorOf<[F32]>]>
    default_float = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "-0.0">
    default_int64 = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "-1">
    default_string = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "_Unused">
    keys_floats = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    keys_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    keys_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    values_floats = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    values_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    values_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>, TensorOf<[F32]>]>


@irdl_op_definition
class ONNXLinearClassifierOp(Operation):
    name: str = "onnx.LinearClassifier"
    summary: str = \
r"""ONNX LinearClassifier operation"""
    description: str =\
r"""
Linear classifier
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    classlabels_ints = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    classlabels_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    coefficients = OptAttributeDef(
        Attribute)  # should actually be F32ArrayAttr
    intercepts = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    multi_class = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    post_transform = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>]>
    Z = ResultDef(Attribute)  # should actually be TensorOf<[F32]>


@irdl_op_definition
class ONNXLinearRegressorOp(Operation):
    name: str = "onnx.LinearRegressor"
    summary: str = \
r"""ONNX LinearRegressor operation"""
    description: str =\
r"""
Generalized linear regression evaluation.<br>
If targets is set to 1 (default) then univariate regression is performed.<br>
If targets is set to M then M sets of coefficients must be passed in as a sequence
and M results will be output for each input n in N.<br>
The coefficients array is of length n, and the coefficients for each target are contiguous.
Intercepts are optional but if provided must match the number of targets.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    coefficients = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    intercepts = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    post_transform = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    targets = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXNormalizerOp(Operation):
    name: str = "onnx.Normalizer"
    summary: str = \
r"""ONNX Normalizer operation"""
    description: str =\
r"""
Normalize the input.  There are three normalization modes, which have the corresponding formulas,
defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
Max: Y = X / max(X)<br>
L1:  Y = X / sum(X)<br>
L2:  Y = sqrt(X^2 / sum(X^2)<br>
In all modes, if the divisor is zero, Y == X.
<br>
For batches, that is, N,C tensors, normalization is done along the C axis. In other words, each row
of the batch is normalized independently.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    norm = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "MAX">
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXOneHotEncoderOp(Operation):
    name: str = "onnx.OneHotEncoder"
    summary: str = \
r"""ONNX OneHotEncoder operation"""
    description: str =\
r"""
Replace each input element with an array of ones and zeros, where a single
one is placed at the index of the category that was passed in. The total category count
will determine the size of the extra dimension of the output array Y.<br>
For example, if we pass a tensor with a single value of 4, and a category count of 8,
the output will be a tensor with ``0,0,0,0,1,0,0,0``.<br>
This operator assumes every input feature is from the same set of categories.<br>
If the input is a tensor of float, int32, or double, the data will be cast
to integers and the cats_int64s category list will be used for the lookups.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>, TensorOf<[I32]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    cats_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    cats_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    zeros = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "1">
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXSVMClassifierOp(Operation):
    name: str = "onnx.SVMClassifier"
    summary: str = \
r"""ONNX SVMClassifier operation"""
    description: str =\
r"""
Support Vector Machine classifier
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    classlabels_ints = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    classlabels_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    coefficients = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    kernel_params = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    kernel_type = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "LINEAR">
    post_transform = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    prob_a = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    prob_b = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    rho = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    support_vectors = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    vectors_per_class = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>]>
    Z = ResultDef(Attribute)  # should actually be TensorOf<[F32]>


@irdl_op_definition
class ONNXSVMRegressorOp(Operation):
    name: str = "onnx.SVMRegressor"
    summary: str = \
r"""ONNX SVMRegressor operation"""
    description: str =\
r"""
Support Vector Machine regression prediction and one-class SVM anomaly detection.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    coefficients = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    kernel_params = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    kernel_type = OptAttributeDef(
        Attribute
    )  # should actually be DefaultValuedStrAttr<StrAttr, "LINEAR">
    n_supports = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    one_class = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<SI64Attr, "0">
    post_transform = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    rho = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    support_vectors = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXScalerOp(Operation):
    name: str = "onnx.Scaler"
    summary: str = \
r"""ONNX Scaler operation"""
    description: str =\
r"""
Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    offset = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    scale = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXTreeEnsembleClassifierOp(Operation):
    name: str = "onnx.TreeEnsembleClassifier"
    summary: str = \
r"""ONNX TreeEnsembleClassifier operation"""
    description: str =\
r"""
Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.<br>
Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.<br>
One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    base_values = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    class_ids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    class_nodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    class_treeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    class_weights = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    classlabels_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    classlabels_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    nodes_falsenodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_featureids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_hitrates = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    nodes_missing_value_tracks_true = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_modes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    nodes_nodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_treeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_truenodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_values = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    post_transform = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I64]>]>
    Z = ResultDef(Attribute)  # should actually be TensorOf<[F32]>


@irdl_op_definition
class ONNXTreeEnsembleRegressorOp(Operation):
    name: str = "onnx.TreeEnsembleRegressor"
    summary: str = \
r"""ONNX TreeEnsembleRegressor operation"""
    description: str =\
r"""
Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.<br>
All fields prefixed with target_ are tuples of votes at the leaves.<br>
A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.<br>
All trees must have their node ids start at 0 and increment by 1.<br>
Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>]>
    aggregate_function = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "SUM">
    base_values = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    n_targets = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<SI64Attr>
    nodes_falsenodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_featureids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_hitrates = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    nodes_missing_value_tracks_true = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_modes = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    nodes_nodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_treeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_truenodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    nodes_values = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    post_transform = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedStrAttr<StrAttr, "NONE">
    target_ids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    target_nodeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    target_treeids = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    target_weights = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<F32ArrayAttr>
    Y = ResultDef(Attribute)  # should actually be outs TensorOf<[F32]>


@irdl_op_definition
class ONNXZipMapOp(Operation):
    name: str = "onnx.ZipMap"
    summary: str = \
r"""ONNX ZipMap operation"""
    description: str =\
r"""
Creates a map from the input and the attributes.<br>
The values are provided by the input tensor, while the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
"""
    X = OperandDef(Attribute)  # should actually be TensorOf<[F32]>
    classlabels_int64s = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<I64ArrayAttr>
    classlabels_strings = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    Z = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[SeqOf<[TupleOf<[StringType, F32]>]>, SeqOf<[TupleOf<[I64, F32]>]>]>


@irdl_op_definition
class ONNXAdagradOp(Operation):
    name: str = "onnx.Adagrad"
    summary: str = \
r"""ONNX Adagrad operation"""
    description: str =\
r"""
Compute one iteration of ADAGRAD, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.
"""
    R = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>
    T = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>
    decay_factor = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    epsilon = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    norm_coefficient = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>


@irdl_op_definition
class ONNXAdamOp(Operation):
    name: str = "onnx.Adam"
    summary: str = \
r"""ONNX Adam operation"""
    description: str =\
r"""
Compute one iteration of Adam, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.
"""
    R = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>
    T = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>
    alpha = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.9">
    beta = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.999">
    epsilon = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    norm_coefficient = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    norm_coefficient_post = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.0">
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>


@irdl_op_definition
class ONNXGradientOp(Operation):
    name: str = "onnx.Gradient"
    summary: str = \
r"""ONNX Gradient operation"""
    description: str =\
r"""
Gradient operator computes the partial derivatives of a specific tensor w.r.t.
some other tensors. This operator is widely used in gradient-based training
algorithms. To illustrate its use, let's consider a computation graph,
"""
    Inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>
    xs = OptAttributeDef(Attribute)  # should actually be StrArrayAttr
    y = OptAttributeDef(Attribute)  # should actually be StrAttr
    zs = OptAttributeDef(
        Attribute)  # should actually be OptionalAttr<StrArrayAttr>
    Outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>


@irdl_op_definition
class ONNXMomentumOp(Operation):
    name: str = "onnx.Momentum"
    summary: str = \
r"""ONNX Momentum operation"""
    description: str =\
r"""
Compute one iteration of stochastic gradient update with momentum.
This operator can conduct the optimization of multiple tensor variables.
"""
    R = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>
    T = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
    inputs = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>
    alpha = OptAttributeDef(Attribute)  # should actually be F32Attr
    beta = OptAttributeDef(Attribute)  # should actually be F32Attr
    mode = OptAttributeDef(Attribute)  # should actually be StrAttr
    norm_coefficient = OptAttributeDef(Attribute)  # should actually be F32Attr
    outputs = ResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>
