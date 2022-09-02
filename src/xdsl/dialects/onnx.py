# This file was generated using the script src/tools/tablegen_to_irdl.py. Editing it is a bad idea.
from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *


@dataclass
class Onnx:
    ctx: MLContext

    def __post_init__(self):
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
        self.ctx.register_op(ONNXBitShiftOp)
        self.ctx.register_op(ONNXCastOp)
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
        self.ctx.register_op(ONNXHardSigmoidOp)
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

 ```
 * pad_shapei is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shapei = ceil((input_spatial_shapei - kernel_spatial_shapei + 1) / strides_spatial_shapei)
 SAME_UPPER or SAME_LOWER: output_spatial_shapei = ceil(input_spatial_shapei / strides_spatial_shapei)
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shapei = (output_spatial_shapei - 1) * strides_spatial_shapei + kernel_spatial_shapei - input_spatial_shapei
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 
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
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See the doc(IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    scale = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    mean = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    var = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    epsilon = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "1e-05">
    momentum = OptAttributeDef(
        Attribute)  # should actually be DefaultValuedAttr<F32Attr, "0.9">
    Y = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    out_mean = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    out_var = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    saved_mean = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>
    saved_var = ResultDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>


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

 Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
 not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

Casting from string tensor in plain (e.g., \3.14\ and \1000\) and scientific numeric representations
(e.g., \1e-5\ and \1E8\) to float types is supported. For example, converting string \100.5\ to an integer may
result 100. There are some string literals reserved for special floating-point values
\+INF\ (and \INF\), \-INF\, and \NaN\ are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match \+INF\ in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to \INF\ and \NaN\. When casting from numeric tensors
to string tensors, plain floating-point representation (such as \314.15926\) would be used.
Converting non-numerical-literal string such as \Hello World!\ is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as \2.718\, to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.
"""
    input = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>
    to = OptAttributeDef(Attribute)  # should actually be TypeAttr
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

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
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

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shapei = stridei * (input_sizei - 1) + output_paddingi + ((kernel_shapei - 1) * dilationsi + 1) - padsstart_i - padsend_i

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_paddingi = stridei * (input_sizei - 1) + output_paddingi + ((kernel_shapei - 1) * dilationsi + 1) - output_shapei
  If (auto_pads == SAME_UPPER): padsstart_i = total_paddingi/2 padsend_i = total_paddingi - (total_paddingi/2)
  Else: padsstart_i = total_paddingi - (total_paddingi/2) padsend_i = (total_paddingi/2).

    
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

Example:
```
input_x = 1, 2, 3
axis=0
output = 1, 3, 6
exclusive=1
output = 0, 1, 3
exclusive=0
reverse=1
output = 6, 5, 3
exclusive=1
reverse=1
output = 5, 3, 0
```
 
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

b, c, h, w = x.shape

tmp = np.reshape(x, b, blocksize, blocksize, c // (blocksize**2), h, w)

tmp = np.transpose(tmp, 0, 3, 4, 1, 5, 2)

y = np.reshape(tmp, b, c // (blocksize**2), h * blocksize, w * blocksize)


In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, b, c // (blocksize ** 2), blocksize, blocksize, h, w)

tmp = np.transpose(tmp, 0, 1, 4, 2, 5, 3)

y = np.reshape(tmp, b, c // (blocksize ** 2), h * blocksize, w * blocksize)

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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
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
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See the doc(IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
"""
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

```outputoutput-term = reduce-sum( input1term1 * input2term )```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by \->\ to separate the left and right hand side of the equation.
If the equation contains \->\ followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis (\...\) to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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
Dimensions are right alignment
Two corresponding dimension must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
"""
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

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`Wzrh` - W parameter weight matrix for update, reset, and hidden gates

`Rzrh` - R recurrence weight matrix for update, reset, and hidden gates

`Wbzrh` - W bias vectors for update, reset, and hidden gates

`Rbzrh` - R bias vectors for update, reset, and hidden gates

`WBzrh` - W parameter weight matrix for backward update, reset, and hidden gates

`RBzrh` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBbzrh` - W bias vectors for backward update, reset, and hidden gates

`RBbzrh` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^-2x)/(1 + e^-2x)

  Sigmoid(x)             - 1/(1 + e^-x)

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See the doc(IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
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

axis = 0 :

Let
k = indicesi_0, ..., i_q-1\\
Then
outputi_0, ..., i_q-1, j_0, ..., j_r-2\\ = inputk , j_0, ..., j_r-2\\

```
  data = 
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  
  indices = 
      0, 1,
      1, 2,
  
  output = 
      
          1.0, 1.2,
          2.3, 3.4,
      ,
      
          2.3, 3.4,
          4.5, 5.7,
      ,
  
```
axis = 1 :

Let
k = indicesi_0, ..., i_q-1\\
Then
outputi_0, ..., i_q-1, j_0, ..., j_r-2\\ = inputj_0, k, j_1, ..., j_r-2\\

```
  data = 
      1.0, 1.2, 1.9,
      2.3, 3.4, 3.9,
      4.5, 5.7, 5.9,
  
  indices = 
      0, 2,
  
  axis = 1,
  output = 
          1.0, 1.9,
          2.3, 3.9,
          4.5, 5.9,
  
```
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

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
  outijk = inputindexijkjk if axis = 0,
  outijk = inputiindexijkk if axis = 1,
  outijk = inputijindexijk if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
  data = 
      1, 2,
      3, 4,
  
  indices = 
      0, 0,
      1, 0,
  
  axis = 1
  output = 
      1, 1,
      4, 3,
  
```
Example 2:
```
  data = 
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
  
  indices = 
      1, 2, 0,
      2, 0, 0,
  
  axis = 0
  output = 
      4, 8, 3,
      7, 2, 3,
  
```
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

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape-1` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds -s, s-1 along axis of size `s` (i.e.) `-data_shapei <= indices...,i <= data_shapei - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape-1 > r-b` => error condition

2) If `indices_shape-1 == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data0:b-1,indices_slice`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape-1 < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data0:b-1, indices_slice , :` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  batch_dims = 0

  data    = 0,1,2,3   # data_shape = 2, 2

  indices = 0,0,1,1   # indices_shape = 2, 2

  output  = 0,3           # output_shape = 2

`Example 2`

  batch_dims = 0

  data    = 0,1,2,3  # data_shape = 2, 2

  indices = 1,0      # indices_shape = 2, 1

  output  = 2,3,0,1  # output_shape = 2, 2

`Example 3`

  batch_dims = 0

  data    = 0,1,2,3,4,5,6,7 # data_shape = 2, 2, 2

  indices = 0,1,1,0                 # indices_shape = 2, 2

  output  = 2,3,4,5                 # output_shape = 2, 2

`Example 4`

  batch_dims = 0

  data    = 0,1,2,3,4,5,6,7 # data_shape = 2, 2, 2

  indices = 0,1,1,0             # indices_shape = 2, 1, 2

  output  = 2,3,4,5             # output_shape = 2, 1, 2

`Example 5`

  batch_dims = 1

  data    = 0,1,2,3,4,5,6,7 # data_shape = 2, 2, 2

  indices = 1,0             # indices_shape = 2, 1

  output  = 2,3,4,5             # output_shape = 2, 2


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

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B) for more details please check the doc(Broadcasting.md).
This operator has **optional** inputs/outputs. See the doc(IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
"""
    A = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    B = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>
    C = ResultDef(Attribute)  # should actually be outs TensorOf<[I1]>


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
class ONNXHardmaxOp(Operation):
    name: str = "onnx.Hardmax"
    summary: str = \
r"""ONNX Hardmax operation"""
    description: str =\
r"""
The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The \axis\ attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.
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
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    output = ResultDef(
        Attribute
    )  # should actually be outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>


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
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>


@irdl_op_definition
class ONNXInstanceNormalizationOp(Operation):
    name: str = "onnx.InstanceNormalization"
    summary: str = \
r"""ONNX InstanceNormalization operation"""
    description: str =\
r"""
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

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

square_sumn, c, d1, ..., dk = sum(Xn, i, d1, ..., dk ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Yn, c, d1, ..., dk = Xn, c, d1, ..., dk / (bias + alpha / size * square_sumn, c, d1, ..., dk ) ^ beta
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

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`Wiofc` - W parameter weight matrix for input, output, forget, and cell gates

`Riofc` - R recurrence weight matrix for input, output, forget, and cell gates

`Wbiofc` - W bias vectors for input, output, forget, and cell gates

`Rbiofc` - R bias vectors for input, output, forget, and cell gates

`Piof`  - P peephole weight vector for input, output, and forget gates

`WBiofc` - W parameter weight matrix for backward input, output, forget, and cell gates

`RBiofc` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBbiofc` - W bias vectors for backward input, output, forget, and cell gates

`RBbiofc` - R bias vectors for backward input, output, forget, and cell gates

`PBiof`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^-2x)/(1 + e^-2x)

  Sigmoid(x)             - 1/(1 + e^-x)

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See the doc(IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
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

**History**
- Version 16 adds bfloat16 to the types allowed.
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The \axis\ attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.
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

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input (\\, \\):
        for (int i=0  ++i) 
          cond = ... // Note this value is ignored, but is required in the body
        

    input (\\, cond) // Note this is analogous to a while loop
        bool cond = ...
        for (int i=0 cond ++i) 
          cond = ...
        

    input (\\, 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0 cond ++i) 
          cond = ...
        

    input (trip_count, \\) // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0 i < trip_count ++i) 
          cond = ... // ignored
        

    input (trip_count, cond)
        int trip_count = ...
        bool cond = ...
        for (int i=0 i < trip_count && cond ++i) 
          cond = ...
        


*Sample usage - cond as well as trip count*

    graph predict-net 
      %a = Constantvalue = <Scalar Tensor 3>()
      %b = Constantvalue = <Scalar Tensor 6>()
      %keepgoing = Constantvalue = <Scalar Tensor 1>()
      %max_trip_count = Constantvalue = <Scalar Tensor 10>()
      %keepgoing_out, %b_out, %user_defined_vals = Loopbody = <graph body-net>(%max_trip_count, %keepgoing, %b)
      return
    

    graph body-net (
      %iINT32, scalar           // iteration number
      %keepgoing_inBOOL, scalar // incoming loop-termination-condition not used
      %b_inINT32, scalar        // incoming value of loop-carried-dependency b
    ) 
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    

*Sample equivalent C code*

    
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6
      bool keepgoing = true // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10 // Analogous to input M
      int user_defined_vals // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0 i < max_trip_count && keepgoing_out ++i) 
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out
        bool b_in = b_out

        /* User-defined code (loop body) */
        int my_local = a + b_in // Reading value \a\ from the enclosing scope is fine
        b_out = a - b_in
        keepgoing_out = my_local > b_out
        user_defined_val = b_in + b_in // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_valsi = user_defined_val // accumulate scan-output values
      
      // int t = my_local // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out user_defined_vals keepgoing_out
    

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable \a\ here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support \diagonal\ or \wavefront\ execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
"""
    M = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I64]>, NoneType]>
    cond = OperandDef(
        Attribute)  # should actually be AnyTypeOf<[TensorOf<[I1]>, NoneType]>
    v_initial = VarOperandDef(
        Attribute
    )  # should actually be Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>
    v_final_and_scan_outputs = VarResultDef(
        Attribute
    )  # should actually be outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>


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

 ```
 * pad_shapei is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shapei = ceil((input_spatial_shapei - ((kernel_spatial_shapei - 1) * dilationsi + 1) + 1) / strides_spatial_shapei)
 SAME_UPPER or SAME_LOWER: output_spatial_shapei = ceil(input_spatial_shapei / strides_spatial_shapei)
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shapei = (output_spatial_shapei - 1) * strides_spatial_shapei + ((kernel_spatial_shapei - 1) * dilationsi + 1) - input_spatial_shapei
 ```
 The output of each pooling window is maximum number of elements exclude pad. 
 
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

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.
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

    Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
    (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
    This attribute is set to 0 by default causing the behavior to be like integer mod.
    Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

    If the input type is floating point, then `fmod` attribute must be set to 1.

    In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
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

    lossnd_1d_2...d_k = -inputncd_1d_2...d_k.

When an optional \weight\ is provided, the sample loss is calculated as:

    lossnd_1d_2...d_k = -inputncd_1d_2...d_k * weightc.

loss is zero for the case when target-value equals ignore_index.

    lossnd_1d_2...d_k = 0, when targetnd_1d_2...d_k = ignore_index

If \reduction\ attribute is set to \none\, the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If \reduction\ attribute is set to \mean\ (the default attribute value), the output loss is (weight) averaged:

    mean(loss), if \weight\ is not provided,

or if weight is provided,

    sum(loss) / sum(weighttargetnd_1d_2...d_k), for all samples.

If \reduction\ attribute is set to \sum\, the output is a scalar:
    sum(loss).

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

    // negative log likelihood loss, \none\ reduction
    N, C, d1 = 2, 3, 2
    input = 1.0, 2.0, 2.0, 2.0, 3.0, 2.0,
             0.0, 1.0, 2.0, 2.0, 1.0, 2
    target = 2, 1, 0, 2

    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = targetnd_1
            lossnd_1 = -inputncd_1

    // print(loss)
    // -3. -2.
    //  -0. -2.

Example 2:

    // weighted negative log likelihood loss, sum reduction
    N, C, d1 = 2, 3, 2
    input = 1.0, 2.0, 2.0, 2.0, 3.0, 2.0,
            0.0, 1.0, 2.0, 2.0, 1.0, 2
    target = 2, 1, 0, 2
    weight = 0.2, 0.3, 0.1
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = targetnd_1
            lossnd_1 = -inputncd_1 * weightc

    loss = np.sum(loss)
    // print(loss)
    // -1.1

Example 3:

    // weighted negative log likelihood loss, mean reduction
    N, C, d1 = 2, 3, 2
    input = 1.0, 2.0, 2.0, 2.0, 3.0, 2.0,
            0.0, 1.0, 2.0, 2.0, 1.0, 2
    target = 2, 1, 0, 2
    weight = 0.2, 0.3, 0.1
    loss = np.zeros((N, d1))
    weight_total = 0
    for n in range(N):
        for d_1 in range(d1):
            c = targetnd_1
            lossnd_1 = -inputncd_1 * weightc
            weight_total = weight_total + weightc

    loss = np.sum(loss) / weight_total
    // print(loss)
    // -1.57
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

    when axis = 0:
    outputinputi, j, k, i, j, k = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    outputi, j, k, inputi, j, k = 1 for all i, j, k and 0 otherwise.

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
class ONNXOrOp(Operation):
    name: str = "onnx.Or"
    summary: str = \
r"""ONNX Or operation"""
    description: str =\
r"""
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

**History**
- Version 16 adds bfloat16 to the types allowed.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X) for more details please check the doc(Broadcasting.md).
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

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  

  pads = 0, 2, 0, 0

  mode = 'constant'

  constant_value = 0.0

  output =
  
      0.0, 0.0, 1.0, 1.2,
      0.0, 0.0, 2.3, 3.4,
      0.0, 0.0, 4.5, 5.7,
  


Example 2 (`reflect` mode):
  data =
  
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  

  pads = 0, 2, 0, 0

  mode = 'reflect'

  output =
  
      1.0, 1.2, 1.0, 1.2,
      2.3, 3.4, 2.3, 3.4,
      4.5, 5.7, 4.5, 5.7,
  


Example 3 (`edge` mode):
  data =
  
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  

  pads = 0, 2, 0, 0

  mode = 'edge'

  output =
  
      1.0, 1.0, 1.0, 1.2,
      2.3, 2.3, 2.3, 3.4,
      4.5, 4.5, 4.5, 5.7,
  

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

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  

  pads = 0, 2, 0, 0

  mode = 'constant'

  constant_value = 0.0

  output =
  
      0.0, 0.0, 1.0, 1.2,
      0.0, 0.0, 2.3, 3.4,
      0.0, 0.0, 4.5, 5.7,
  


Example 2 (`reflect` mode):
  data =
  
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  

  pads = 0, 2, 0, 0

  mode = 'reflect'

  output =
  
      1.0, 1.2, 1.0, 1.2,
      2.3, 3.4, 2.3, 3.4,
      4.5, 5.7, 4.5, 5.7,
  


Example 3 (`edge` mode):
  data =
  
      1.0, 1.2,
      2.3, 3.4,
      4.5, 5.7,
  

  pads = 0, 2, 0, 0

  mode = 'edge'

  output =
  
      1.0, 1.0, 1.0, 1.2,
      2.3, 2.3, 2.3, 3.4,
      4.5, 4.5, 4.5, 5.7,
  

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

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^-2x)/(1 + e^-2x)

  Sigmoid(x)             - 1/(1 + e^-x)

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See the doc(IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
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

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
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

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
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

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
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

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
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

The number of elements in the output of range is computed as below-

`number_of_elements = max( ceil( (limit - start) / delta ) , 0 )`

The pseudocode determining the contents of the output is shown below-

`for(int i=0 i<number_of_elements ++i)`

``

`    outputi =  start + (i * delta)  `

``

`Example 1`
Inputs: start = 3, limit = 9, delta = 3
Output: 3, 6

`Example 2`
Inputs: start = 10, limit = 4, delta = -2
Output: 10, 8, 6

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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
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
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
    shape = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
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

For each slice i iterating on batch axis, the operator reverses the first sequence_lensi elements on time axis,
and copies elements whose index's beyond sequence_lensi to the output. So the output slice i contains reversed
sequences on the first sequence_lensi elements, then have original values copied for the other elements.

Example 1:
  input = 0.0, 4.0, 8.0,  12.0,
           1.0, 5.0, 9.0,  13.0,
           2.0, 6.0, 10.0, 14.0,
           3.0, 7.0, 11.0, 15.0
  sequence_lens = 4, 3, 2, 1
  time_axis = 0
  batch_axis = 1

  output = 3.0, 6.0, 9.0,  12.0,
            2.0, 5.0, 8.0,  13.0,
            1.0, 4.0, 10.0, 14.0,
            0.0, 7.0, 11.0, 15.0

Example 2:
  input = 0.0,  1.0,  2.0,  3.0 ,
           4.0,  5.0,  6.0,  7.0 ,
           8.0,  9.0,  10.0, 11.0,
           12.0, 13.0, 14.0, 15.0
  sequence_lens = 1, 2, 3, 4
  time_axis = 1
  batch_axis = 0

  output = 0.0,  1.0,  2.0,  3.0 ,
            5.0,  4.0,  6.0,  7.0 ,
            10.0, 9.0,  8.0,  11.0,
            15.0, 14.0, 13.0, 12.0
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

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
"""
    X = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    rois = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>
    batch_indices = OperandDef(Attribute)  # should actually be TensorOf<[I64]>
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

Examples:
```
round(0.9) = 1.0
round(2.5) = 2.0
round(2.3) = 2.0
round(1.5) = 2.0
round(-4.5) = -4.0
```
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

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = axis_1, ..., axis_m
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shapeaxis_i denotes the (max) sequence-length of scan_i
    // scan_i.shapeaxis_i is required to be equal to scan_j.shapeaxis_j for all i,j.
    sequence_length = scan_1.shapeaxis_1

    // initialize state-variables
    st_1 = init_1 ... st_n = init_n
    // initialize scan-output variables:  denotes an empty tensor
    scan_out_1 =  ... scan_out_k = 
    // identify number of iterations:

    // execute loop
    for (int t = 0 t < sequence_length ++t) 
        // generate the scan-input elements: the notation T<axis=k>t indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>t
        ... 
        si_m = scan_m<axis=axis_m>t
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1) ...  scan_out_k = Concat<axis=0>(scan_out_k, so_k)
    

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding 
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scanbody = <graph rnn-cell-1>, num_scan_inputs=1(%H_0, %X)
      return %Y, %Y_h
    

    graph rnn-cell-1 (
      %H_tminus1FLOAT, tensor
      %X_tFLOAT, tensor
    ) 
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    

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

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the ij entry
is performed as below:
```
  outputindicesijj = updatesij if axis = 0,
  outputiindicesij = updatesij if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = 
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
  
  indices = 
      1, 0, 2,
      0, 2, 1,
  
  updates = 
      1.0, 1.1, 1.2,
      2.0, 2.1, 2.2,
  
  output = 
      2.0, 1.1, 0.0
      1.0, 0.0, 2.2
      0.0, 2.1, 1.2
  
```
Example 2:
```
  data = 1.0, 2.0, 3.0, 4.0, 5.0
  indices = 1, 3
  updates = 1.1, 2.1
  axis = 1
  output = 1.0, 1.1, 3.0, 2.1, 5.0
```
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

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the ij entry
is performed as below:
```
  outputindicesijj = updatesij if axis = 0,
  outputiindicesij = updatesij if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = 
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
  
  indices = 
      1, 0, 2,
      0, 2, 1,
  
  updates = 
      1.0, 1.1, 1.2,
      2.0, 2.1, 2.2,
  
  output = 
      2.0, 1.1, 0.0
      1.0, 0.0, 2.2
      0.0, 2.1, 1.2
  
```
Example 2:
```
  data = 1.0, 2.0, 3.0, 4.0, 5.0
  indices = 1, 3
  updates = 1.1, 2.1
  axis = 1
  output = 1.0, 1.1, 3.0, 2.1, 5.0
```
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

`indices` is an integer tensor. Let k denote indices.shape-1, the last dimension in the shape of `indices`.
 `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape0:q-1 ++ data.shapek:r-1, where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

    output = np.copy(data)
    update_indices = indices.shape:-1
    for idx in np.ndindex(update_indices):
        outputindicesidx = updatesidx

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indicesidx1 != indicesidx2.
This ensures that the output value does not depend on the iteration order.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to \none\, indices should not have duplicate entries: that is, if idx1 != idx2, 
then indicesidx1 != indicesidx2. This ensures that the output value does not depend on the iteration order.
When `reduction` is set to \add\, `output` is calculated as follows:

    output = np.copy(data)
    update_indices = indices.shape:-1
    for idx in np.ndindex(update_indices):
        outputindicesidx += updatesidx

When `reduction` is set to \mul\, `output` is calculated as follows:

    output = np.copy(data)
    update_indices = indices.shape:-1
    for idx in np.ndindex(update_indices):
        outputindicesidx *= updatesidx

This operator is the inverse of GatherND.

Example 1:
```
  data    = 1, 2, 3, 4, 5, 6, 7, 8
  indices = 4, 3, 1, 7
  updates = 9, 10, 11, 12
  output  = 1, 11, 3, 10, 9, 6, 7, 12
```

Example 2:
```
  data    = 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
             1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
             8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
             8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8
  indices = 0, 2
  updates = 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
             1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4
  output  = 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
             1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
             1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
             8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8
```
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
"""
    data = OperandDef(
        Attribute
    )  # should actually be AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>
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
bias. The formula of this operator is: If x < -lambd, y = x + bias
If x > lambd, y = x - bias Otherwise, y = 0.
"""
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

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 

The \axis\ attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.
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

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labelsi <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    lid1d2...dk = -yicd1d2..dk, where i is the index of classes.
or
    lid1d2...dk = -yicd1d2..dk * weightsc, if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    lid1d2...dk  = 0, when labelsnd1d2...dk = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labelsid1d2...dk

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and Wnd1d2...dk = weightslabelsid1d2...dk.
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
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

In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
Let's consider an example. Assume that input sequence is 94, 17, 36, 12, 28 and the number of skips is 2.
The associated 2-grams are 94, 12 and 17, 28 respectively indexed by 0, 3 and 1, 4.
If the number of skips becomes 0, the 2-grams generated are 94, 17, 17, 36, 36, 12, 12, 28
indexed by 0, 1, 1, 2, 2, 3, 3, 4, respectively.

The output vector (denoted by Y) stores the count of each n-gram
Yngram_indexesi indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
between index i and the corresponding n-gram's output coordinate. If pool_int64s is 94, 17, 17, 36, ngram_indexes is 1, 0,
ngram_counts=0, 0, then the Y0 (first element in Y) and Y1 (second element in Y) are the counts of 17, 36 and 94, 17,
respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
Note that we may consider all skips up to S when generating the n-grams.

The examples used above are true if mode is \TF\. If mode is \IDF\, all the counts larger than 1 would be truncated to 1 and
the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is \TFIDF\,
this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
If pool_strings is set, the input must be a string tensor.
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

If \largest\ is 1 (the default value) then the k largest elements are returned.
If \sorted\ is 1 (the default value) then the resulting k elements will be sorted.
If \sorted\ is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
 a tiebreaker. That is, the element with the lower index will appear first.
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
class ONNXUniqueOp(Operation):
    name: str = "onnx.Unique"
    summary: str = \
r"""ONNX Unique operation"""
    description: str =\
r"""
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'..
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. \.
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
  input_X = 2, 1, 1, 3, 4, 3
  attribute_sorted = 0
  attribute_axis = None
  output_Y = 2, 1, 3, 4
  output_indices = 0, 1, 3, 4
  output_inverse_indices = 0, 1, 1, 2, 3, 2
  output_counts = 1, 2, 2, 1

Example 2:
  input_X = 1, 3, 2, 3
  attribute_sorted = 1
  attribute_axis = None
  output_Y = 1, 2, 3
  output_indices = 0, 2, 1
  output_inverse_indices = 0, 2, 1, 2
  output_counts = 1, 1, 2

Example 3:
  input_X = 1, 0, 0, 1, 0, 0, 2, 3, 4
  attribute_sorted = 1
  attribute_axis = 0
  output_Y = 1, 0, 0, 2, 3, 4
  output_indices = 0, 2
  output_inverse_indices = 0, 0, 1
  output_counts = 2, 1

Example 4:
  input_x = 1., 1., 0., 1., 2., 1., 0., 1.,
             1., 1., 0., 1., 2., 1., 0., 1.
  attribute_sorted = 1
  attribute_axis = 1

  intermediate data are presented below for better understanding:

  there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  A: 1, 1, 1, 1,
     0, 1, 0, 1,
     2, 1, 2, 1,
     0, 1, 0, 1.

  there are 3 unique subtensors:
  1, 1, 1, 1,
  0, 1, 0, 1,
  2, 1, 2, 1.

  sorted unique subtensors:
  B: 0, 1, 0, 1,
     1, 1, 1, 1,
     2, 1, 2, 1.

  output_Y is constructed from B:
  0. 1., 1. 1., 2. 1.,
   0. 1., 1. 1., 2. 1.

  output_indices is to map from B to A:
  1, 0, 2

  output_inverse_indices is to map from A to B:
  1, 0, 2, 0

  output_counts = 2 1 1
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

For example:
  Given an input tensor (`data`) of shape 3, 4, 5, then
  Unsqueeze(data, axes=0, 4) outputs a tensor (`expanded`) containing same data as `data` but with shape 1, 3, 4, 5, 1.

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range -output_rank , output_rank - 1.
The order of values in `axes` does not matter and can come in any order.

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

For example:
  Given an input tensor (`data`) of shape 3, 4, 5, then
  Unsqueeze(data, axes=0, 4) outputs a tensor (`expanded`) containing same data as `data` but with shape 1, 3, 4, 5, 1.

The attribute `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range -output_rank , output_rank - 1.
The order of values in `axes` does not matter and can come in any order.

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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).

**History**
- Version 16 adds bfloat16 to the types allowed (for the second and third parameter).
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

This operator supports **multidirectional (i.e., Numpy-style) broadcasting** for more details please check the doc(Broadcasting.md).
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

    Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    some parameters:

     - The initial learning-rate \R\.
     - The update count \T\. That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient \norm_coefficient\.
     - A learning-rate decay factor \decay_factor\.
     - A small constant \epsilon\ to avoid dividing-by-zero.

    At each ADAGRAD iteration, the optimized tensors are moved along a direction
    computed based on their estimated gradient and accumulated squared gradient. Assume
    that only a single tensor \X\ is updated by this operator. We need the value of \X\,
    its gradient \G\, and its accumulated squared gradient \H\. Therefore, variables in
    this operator's input list are sequentially \R\, \T\, \X\, \G\, and \H\. Other
    parameters are given as attributes because they are usually constants. Also, the
    corresponding output tensors are the new value of \X\ (called \X_new\), and then
    the new accumulated squared gradient (called \H_new\). Those outputs are computed
    from the given inputs following the pseudo code below.

    Let \+\, \-\, \*\, and \/\ are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Compute a scalar learning-rate factor. At the first update of X, T is generally
      // 0 (0-based update index) or 1 (1-based update index).
      r = R / (1 + T * decay_factor)

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G

      // Compute new accumulated squared gradient.
      H_new = H + G_regularized * G_regularized

      // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
      // computes element-wise square-root.
      H_adaptive = Sqrt(H_new) + epsilon

      // Compute the new value of \X\.
      X_new = X - r * G_regularized / H_adaptive

    If one assign this operators to optimize multiple inputs, for example, \X_1\ and \X_2\, the same
    pseudo code may be extended to handle all tensors jointly. More specifically, we can view \X\ as a
    concatenation of \X_1\ and \X_2\ (of course, their gradient and accumulate gradient should
    be concatenated too) and then just reuse the entire pseudo code.

    Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    In that reference paper, this operator is a special case of the Figure 1's composite mirror
    descent update.
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

    Let's define the behavior of this operator. First of all, Adam requires
    some parameters:

     - The learning-rate \R\.
     - The update count \T\. That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient \norm_coefficient\.
     - A small constant \epsilon\ to avoid dividing-by-zero.
     - Two coefficients, \alpha\ and \beta\.

    At each Adam iteration, the optimized tensors are moved along a direction
    computed based on their exponentially-averaged historical gradient and
    exponentially-averaged historical squared gradient. Assume that only a tensor
    \X\ is being optimized. The rest of required information is

     - the value of \X\,
     - \X\'s gradient (denoted by \G\),
     - \X\'s exponentially-averaged historical gradient (denoted by \V\), and
     - \X\'s exponentially-averaged historical squared gradient (denoted by \H\).

    Some of those parameters are passed into this operator as input tensors and others
    are stored as this operator's attributes. Specifically, this operator's input tensor
    list is \R\, \T\, \X\, \G\, \V\, \H\. That is, \R\ is the first input, \T\ is
    the second input, and so on. Other parameters are given as attributes because they
    are constants. Moreover, the corresponding output tensors are

     - the new value of \X\ (called \X_new\),
     - the new exponentially-averaged historical gradient (denoted by \V_new\), and
     - the new exponentially-averaged historical squared gradient (denoted by \H_new\).

    Those outputs are computed following the pseudo code below.

    Let \+\, \-\, \*\, and \/\ are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G

      // Update exponentially-averaged historical gradient.
      V_new = alpha * V + (1 - alpha) * G_regularized

      // Update exponentially-averaged historical squared gradient.
      H_new = beta * H + (1 - beta) * G_regularized * G_regularized

      // Compute the element-wise square-root of H_new. V_new will be element-wisely
      // divided by H_sqrt for a better update direction.
      H_sqrt = Sqrt(H_new) + epsilon

      // Compute learning-rate. Note that \alpha**T\/\beta**T\ is alpha's/beta's T-th power.
      R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

      // Compute new value of \X\.
      X_new = X - R_adjusted * V_new / H_sqrt

      // Post-update regularization.
      X_final = (1 - norm_coefficient_post) * X_new

    If there are multiple inputs to be optimized, the pseudo code will be applied
    independently to each of them.
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

```
X -----.
       |
       v
W --> Conv --> H --> Gemm --> Y
                      ^
                      |
                      Z
```

, where W and Z are trainable tensors. Note that operators' attributes are
omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
Y with respect to W (Z). The user can compute gradient by inserting Gradient
operator to form another graph shown below.

```
W --> Conv --> H --> Gemm --> Y
|      ^              ^
|      |              |
|      X              Z
|      |              |
|      |   .----------'
|      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
|      |   |   \xs\ followed by \zs\)
|      v   v
'---> Gradient(xs=\W\, \Z\, zs=\X\, y=\Y\)
       |   |
       |   '-----------------------------------> dY/dW (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

By definition, the tensor \y\ is a function of independent variables in \xs\
and \zs\. Since we only compute the gradient of \y\ w.r.t. the differentiable
variables in \xs\, this Gradient only outputs dY/dW and dY/dZ. Note that \H\
cannot appear in \xs\ and \zs\. The reason is that \H\ can be determined by
tensors \W\ and \X\ and therefore \H\ is not an independent variable.

All outputs are optional. If needed, for example, user can assign an empty
string to the 1st output name of that Gradient to skip the generation of dY/dW.
Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
and LSTM.

Gradient operator can compute derivative against intermediate tensors. For
example, the gradient of Y with respect to H can be done via

```
W --> Conv --> H --> Gemm --> Y
       ^       |      ^
       |       |      |
       X       |      Z
       .-------'      |
       |   .----------'
       |   | (H/Z is the 1st/2nd input of Gradient as shown in \xs\)
       v   v
      Gradient(xs=\H\, \Z\, y=\Y\)
       |   |
       |   '-----------------------------------> dY/dH (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

It is possible to represent high-order differentiation using Gradient operators.
For example, given the following linear model:

```
W --> Gemm --> Y --> Loss --> O
       ^              ^
       |              |
       X              L
```

To compute the 2nd order derivative of O with respect to W (denoted by
d^2O/dW^2), one can do

```
W --> Gemm --> Y --> Loss --> O
|      ^              ^
|      |              |
|      X .------------L
|      | |            |
|      | |            v
+------+-+> Gradient(xs=\X\, \W\, zs=\L\, y=\O\) ---> dO/dX (1st output of Gradient)
|      | |    |
|      | |    '---> dO/dW (2nd output of Gradient)
|      v v
'---> Gradient(xs=\X\, \W\, zs=\L\, y=\dO/dW\) ---> d(dO/dW)dX (1st output of
       |                                                  Gradient)
       |
       |
       '---> d^2O/dW^2 (2nd output of Gradient)
```

The tensors named in attributes \xs\, \zs\, and \y\ define the differentiated
computation graph, and the inputs to Gradient node define the values at
which the gradient is computed. We can feed different tensors to the identified
graph. For example, one can compute the gradient of Y with respect to H at
a specific value of H, H_1, by providing that value as an input to the Gradient
node.

```
W --> Conv --> H --> Gemm --> Y
       ^              ^
       |              |
       X              Z

          Z_1 (2nd input of Gradient)
           |
           v
H_1 --> Gradient(xs=\H\, \Z\, y=\Y\) ---> dY/dH when H = H_1 and Y = Y_1.
           |
           '------------------------------> dY/dZ (2nd output of Gradient)
```

When the inputs of Gradient are the tensors named in \xs\ and \zs\, the
computation can be optimized. More specifically, intermediate variables in
forward pass can be reused if the gradient is computed via reverse-mode
auto-differentiation.

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

    Let's define the behavior of this operator. As you can imagine, SG with momentum requires
    several parameters:

     - The learning-rate \R\.
     - The update count \T\. That is, the number of conducted training iterations. It should
       be zero in the first training iteration.
     - A L2-norm regularization coefficient \norm_coefficient\.
     - A decay coefficient of previous accumulated gradient (i.e., momentum) \alpha\.
     - The scaling coefficient of current gradient \beta\.
     - An attribute to choose either standard momentum or Nesterov's momentum \mode\ should
       be used.

    For the sake of simplicity, assume that there is only one tensor (called \X\) to be optimized.
    Other necessary inputs are \X\'s gradient (called \G\) and \X\'s momentum (called \V\). This
    Momentum operator maps all these inputs to the new value of \X\ (called \X_new\) and its new
    momentum (called \V_new\).

    This operator supports two different momentum algorithms. Set the attribute \mode\ to
    \nesterov\ if Nesterov's momentum is desired. Otherwise, set the attribute \model\ to
    \standard\ to use standard momentum. Computation details are described subsequently.

    Let \+\, \-\, \*\, and \/\ are all element-wise operations with numpy-style broadcasting.

    Pseudo code for SG with standard momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized

      // Update X.
      X_new = X - R * V_new

    Pseudo code for SG with Nesterov's momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized

      // Compute final update direction and then update X.
      X_new = X - R * (G_regularized + alpha * V_new)

    If one assign this operators to optimize multiple inputs, for example, \X_1\ and \X_2\. The same
    pseudo code would be extended to handle all tensors jointly. More specifically, we can view \X\ as a
    concatenation of \X_1\ and \X_2\ (of course, their gradient and accumulate gradient should
    be concatenated too) and then our pseudo code becomes applicable.
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
