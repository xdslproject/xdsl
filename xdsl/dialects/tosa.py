from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I8,
    I32,
    I64,
    AnyAttr,
    AnyFloat,
    AnyFloatConstr,
    BoolAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    ShapedType,
    SignlessIntegerConstraint,
    StringAttr,
    TensorType,
    i1,
)
from xdsl.ir import Attribute, Dialect, SSAValue, TypeAttribute
from xdsl.irdl import (
    AttrConstraint,
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    eq,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    Commutative,
    HasParent,
    IsTerminator,
    Pure,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException


def are_tosa_broadcastable(lhs: Attribute, rhs: Attribute, out: Attribute):
    """
    Returns `True` if lhs and rhs have compatible shapes with broadcasting: the dimensions have to match, or one of them must be
    equal to 1, in which case the corresponding resulting dimension must be equal to the non-1 dimension.

    e.g.: `[1, 2, 3] & [4, 2, 1] -> [4, 2, 3]`
    """
    if (
        not isinstance(lhs, ShapedType)
        or not isinstance(rhs, ShapedType)
        or not isinstance(out, ShapedType)
    ):
        return False

    lhs_shape = lhs.get_shape()
    rhs_shape = rhs.get_shape()
    out_shape = out.get_shape()

    ranks_equal = len(lhs_shape) == len(rhs_shape) == len(out_shape)
    if not ranks_equal:
        return False

    # check that expected dimensions match output dimensions
    # and input dimensions are equal, or broadcast
    return all(
        l == r == o or (l == 1 and r == o) or (r == 1 and l == o)
        for l, r, o in zip(lhs_shape, rhs_shape, out_shape)
    )


@irdl_op_definition
class ClampOp(IRDLOperation):
    """
    Computes clamp(features, min, max)

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosaclamp-mlirtosaclampop)
    """

    name = "tosa.clamp"

    T: ClassVar[VarConstraint] = VarConstraint("T", AnyAttr())
    VALUE: ClassVar[AttrConstraint] = IntegerAttr.constr(
        type=SignlessIntegerConstraint & T
    ) | FloatAttr.constr(type=AnyFloatConstr & T)

    min_val = prop_def(VALUE)
    max_val = prop_def(VALUE)

    nan_mode = opt_prop_def(StringAttr, default_value=StringAttr("PROPAGATE"))

    input = operand_def(TensorType.constr(T))
    output = result_def(TensorType.constr(T))

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($output)"


ROUNDING_MODE_CONSTRAINT = (
    eq(StringAttr("SINGLE_ROUND"))
    | eq(StringAttr("INEXACT_ROUND"))
    | eq(StringAttr("DOUBLE_ROUND"))
)
"""Rounding mode for `tosa.rescale`"""


@irdl_op_definition
class ConstOp(IRDLOperation):
    """
    TOSA const operation.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosaconst-mlirtosaconstop).
    """

    name = "tosa.const"

    values = prop_def(DenseIntOrFPElementsAttr)

    output = result_def(TensorType)

    def __init__(self, values: DenseIntOrFPElementsAttr):
        super().__init__(
            properties={"values": values}, result_types=(values.get_type(),)
        )


@irdl_op_definition
class RescaleOp(IRDLOperation):
    """
    Tosa Rescale Operator

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosarescale-mlirtosarescaleop)
    """

    name = "tosa.rescale"

    scale32 = prop_def(BoolAttr)
    rounding_mode = prop_def(
        ROUNDING_MODE_CONSTRAINT, default_value=StringAttr("PROPAGATE")
    )
    per_channel = prop_def(BoolAttr)
    input_unsigned = prop_def(BoolAttr)
    output_unsigned = prop_def(BoolAttr)

    input = operand_def(TensorType)
    multiplier = operand_def(TensorType)
    shift = operand_def(TensorType)
    input_zp = operand_def(TensorType)
    output_zp = operand_def(TensorType)

    output = result_def(TensorType)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"


class ElementwiseOperation(IRDLOperation, ABC):
    """
    Abstract superclass for elementwise TOSA operations
    """

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    traits = traits_def(
        Pure(),
    )


class ElementwiseBinaryOperation(ElementwiseOperation):
    """
    Abstract superclass for elementwise, binary TOSA operations.
    """

    T: ClassVar = VarConstraint("T", AnyAttr())

    input1 = operand_def(TensorType.constr(T))
    input2 = operand_def(TensorType.constr(T))
    output = result_def(TensorType.constr(T))

    def verify_(self) -> None:
        t1 = self.input1.type
        t2 = self.input2.type
        t_out = self.output.type

        if not are_tosa_broadcastable(t1, t2, t_out):
            raise VerifyException(
                f"'{type(self).name}' Operand and result tensor shapes are not compatible"
            )


@irdl_op_definition
class AddOp(ElementwiseBinaryOperation):
    """
    Tosa elementwise add operation

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosaadd-mlirtosaaddop)
    """

    name = "tosa.add"

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class SubOp(ElementwiseBinaryOperation):
    """
    Tosa elementwise subtraction operation

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosasub-mlirtosasubop)
    """

    name = "tosa.sub"


@irdl_op_definition
class MulOp(ElementwiseOperation):
    """
    Tosa elementwise multiplication operation (Hadamard product)

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosamul-mlirtosamulop)
    """

    name = "tosa.mul"

    traits = traits_def(
        Commutative(),
    )

    T: ClassVar = VarConstraint("T", AnyAttr())

    input1 = operand_def(TensorType.constr(T))
    input2 = operand_def(TensorType.constr(T))
    shift = operand_def(TensorType[I8])
    output = result_def(TensorType.constr(T))

    def verify_(self) -> None:
        t1 = self.input1.type
        t2 = self.input2.type
        t_out = self.output.type

        if not are_tosa_broadcastable(t1, t2, t_out):
            raise VerifyException(
                f"'{type(self).name}' Operand and result tensor shapes are not compatible"
            )


TInv = TypeVar("TInv", bound=TensorType)


class ElementwiseUnaryOperation(ElementwiseOperation, Generic[TInv]):
    """
    Abstract base class for elementwise unary operations on tensors of floating-point types
    """

    input1 = operand_def(TInv)
    result = result_def(TInv)


@irdl_op_definition
class SinOp(ElementwiseUnaryOperation[TensorType[AnyFloat]]):
    """
    TOSA dialect operation computing sin(x) for each element in a tensor

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosasin-mlirtosasinop)
    """

    name = "tosa.sin"


@irdl_op_definition
class CosOp(ElementwiseUnaryOperation[TensorType[AnyFloat]]):
    """
    TOSA dialect operation computing cos(x) for each element in a tensor

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosacos-mlirtosacosop)
    """

    name = "tosa.cos"


@irdl_op_definition
class ReciprocalOp(ElementwiseUnaryOperation[TensorType]):
    """
    Elementwise reciprocal operation.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosareciprocal-mlirtosareciprocalop).
    """

    name = "tosa.reciprocal"


@irdl_op_definition
class MatMulOp(IRDLOperation):
    """
    TOSA dialect operation for computing 2D matmuls. Expects 3D tensors as input with leading rank of 1 element, e.g.

    `tensor<1x14x19xf32> * tensor<1x19x28xf32> -> tensor<1x14x28xf32>`

    The operands `a_zp` and `b_zp` are the zero-point which are used for quantized operations, can be set to 0.0 for no effect.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosamul-mlirtosamulop).
    """

    name = "tosa.matmul"

    T: ClassVar = VarConstraint("T", AnyAttr())

    a = operand_def(TensorType.constr(T))
    b = operand_def(TensorType.constr(T))

    a_zp = operand_def(TensorType.constr(T))
    b_zp = operand_def(TensorType.constr(T))

    output = result_def(TensorType.constr(T))

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    traits = traits_def(
        Pure(),
    )

    def verify_(self) -> None:
        assert isinstance(self.a.type, ShapedType)
        assert isinstance(self.b.type, ShapedType)

        assert isinstance(self.a_zp.type, ShapedType)
        assert isinstance(self.b_zp.type, ShapedType)

        sa = self.a.type.get_shape()
        sb = self.b.type.get_shape()

        s_az = self.a_zp.type.get_shape()
        s_bz = self.b_zp.type.get_shape()

        if len(sa) != 3 or len(sb) != 3:
            raise VerifyException("'tosa.matmul' Expected operand tensors of rank 3")

        if sa[0] != 1 or sb[0] != 1:
            raise VerifyException(
                "'tosa.matmul' Expected leading dimension of input tensors to be 1"
            )

        # expect m x n ... n x k
        if sa[2] != sb[1]:
            raise VerifyException(
                "'tosa.matmul' Incompatible shapes for performing matrix multiplication"
            )

        # check that zero-points are unranked or scalar
        if len(s_az) not in [0, 1] or len(s_bz) not in [0, 1]:
            raise VerifyException(
                "'tosa.matmul' Expected zero-point operands to be unranked or scalar tensors"
            )


@irdl_op_definition
class MaxPool2DOp(IRDLOperation):
    """
    TOSA dialect operation for performing 2D max pooling on a tensor.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosamax_pool2d-mlirtosamaxpool2dop).
    """

    name = "tosa.max_pool2d"

    T: ClassVar = VarConstraint("T", AnyAttr())
    input = operand_def(TensorType.constr(T))
    output = result_def(TensorType.constr(T))

    kernel = prop_def(DenseArrayBase[I64])
    stride = prop_def(DenseArrayBase[I64])
    pad = prop_def(DenseArrayBase[I64])
    nan_mode = opt_prop_def(StringAttr, default_value=StringAttr("PROPAGATE"))

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    def _verify_(self) -> None:
        assert isinstance(self.input.type, ShapedType)
        assert isinstance(self.output.type, ShapedType)

        input_shape = self.input.type.get_shape()
        output_shape = self.output.type.get_shape()

        if len(input_shape) != 4 or len(output_shape) != 4:
            raise VerifyException(
                "'tosa.max_pool2d' Expected input and output tensors to be rank 4"
            )


@irdl_op_definition
class AvgPool2DOp(IRDLOperation):
    """
    TOSA dialect operation for performing 2D average pooling on a tensor.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosaavg_pool2d-mlirtosaavgpool2dop).
    """

    name = "tosa.avg_pool2d"

    input = operand_def(TensorType)
    input_zp = operand_def(TensorType)
    output_zp = operand_def(TensorType)

    kernel = prop_def(DenseArrayBase)
    stride = prop_def(DenseArrayBase)
    pad = prop_def(DenseArrayBase)
    acc_type = prop_def(TypeAttribute)

    output = result_def(TensorType)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    TOSA dialect operation for concatenating tensors.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosaconcat-mlirtosaconcatop).
    """

    name = "tosa.concat"

    tensors = var_operand_def(TensorType)
    axis = prop_def(IntegerAttr[I32])
    output = result_def(TensorType)

    def __init__(
        self, tensors: Sequence[SSAValue], axis: IntegerAttr, output_type: TensorType
    ):
        super().__init__(
            operands=[tensors],
            properties={"axis": axis},
            result_types=[output_type],
        )

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$tensors attr-dict `:` `(` type($tensors) `)` `->` type($output)"


@irdl_op_definition
class YieldOp(IRDLOperation):
    """
    TOSA operation for returning out of conditional and body of structured control flow

    See [external documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosayield-mlirtosayieldop)
    """

    name = "tosa.yield"

    inputs = var_operand_def(TensorType)

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(IfOp),
            Pure(),
        )
    )

    assembly_format = "$inputs attr-dict `:` type($inputs)"


@irdl_op_definition
class IfOp(IRDLOperation):
    """
    Conditional operation on tensors.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/TOSA/#tosacond_if-mlirtosaifop).
    """

    name = "tosa.cond_if"

    cond = operand_def(TensorType(i1, []))

    output = var_result_def(TensorType)

    true_region = region_def("single_block")
    false_region = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        SingleBlockImplicitTerminator(YieldOp),
    )

    assembly_format = (
        "$cond `->` `(` type($output) `)` $true_region `else` $false_region attr-dict"
    )


################################################################################
# Reduction ops                                                                #
################################################################################


class ReductionOperation(IRDLOperation, ABC):
    """
    Base class for all TOSA reduction operations
    """

    input = operand_def(TensorType)
    axis = prop_def(IntegerAttr[I32])

    output = result_def(TensorType)

    assembly_format = "$input attr-dict `:` functional-type(operands, results)"

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class ReduceAllOp(ReductionOperation):
    """
    Reduce a tensor along the given axis with a logical AND operation
    """

    name = "tosa.reduce_all"


@irdl_op_definition
class ReduceAnyOp(ReductionOperation):
    """
    Reduce a tensor along the given axis with a logical OR operation
    """

    name = "tosa.reduce_any"


@irdl_op_definition
class ReduceMaxOp(ReductionOperation):
    """
    Reduce a tensor along the given axis by taking the maximum value
    """

    name = "tosa.reduce_max"

    nan_mode = prop_def(StringAttr, default_value=StringAttr("PROPAGATE"))


@irdl_op_definition
class ReduceMinOp(ReductionOperation):
    """
    Reduce a tensor along the given axis by taking the minimum value
    """

    name = "tosa.reduce_min"

    nan_mode = prop_def(StringAttr, default_value=StringAttr("PROPAGATE"))


@irdl_op_definition
class ReduceProductOp(ReductionOperation):
    """
    Reduce a tensor along the given axis by taking the product of all values
    """

    name = "tosa.reduce_product"


@irdl_op_definition
class ReduceSumOp(ReductionOperation):
    """
    Reduce a tensor along the given axis by taking the product of all values
    """

    name = "tosa.reduce_sum"


TOSA = Dialect(
    "tosa",
    [
        ClampOp,
        ConstOp,
        RescaleOp,
        AddOp,
        SubOp,
        MulOp,
        SinOp,
        CosOp,
        ReciprocalOp,
        ReduceAllOp,
        ReduceAnyOp,
        ReduceMaxOp,
        ReduceMinOp,
        ReduceProductOp,
        ReduceSumOp,
        MatMulOp,
        MaxPool2DOp,
        AvgPool2DOp,
        ConcatOp,
        IfOp,
        YieldOp,
    ],
    [],
)
