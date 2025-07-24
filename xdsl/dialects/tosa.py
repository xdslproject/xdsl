from typing import ClassVar

from xdsl.dialects.builtin import (
    I32,
    I64,
    AnyAttr,
    BoolAttr,
    DenseArrayBase,
    FloatAttr,
    IntegerAttr,
    ShapedType,
    StringAttr,
    TensorType,
)
from xdsl.ir import Attribute, Dialect
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


def are_tosa_broadcastable(t_in1: Attribute, t_in2: Attribute, t_out: Attribute):
    if (
        not isinstance(t_in1, ShapedType)
        or not isinstance(t_in2, ShapedType)
        or not isinstance(t_out, ShapedType)
    ):
        return False

    # check ranks are equal
    if not (t_in1.get_num_dims() == t_in2.get_num_dims() == t_out.get_num_dims()):
        return False

    # check ranks are broadcastable
    in_shapes = zip(t_in1.get_shape(), t_in2.get_shape())

    if not all(dim1 == dim2 or dim1 == 1 or dim2 == 1 for dim1, dim2 in in_shapes):
        return False

    # check output shape is constructed from input shapes
    shapes = zip(t_in1.get_shape(), t_in2.get_shape(), t_out.get_shape())
    return all(dim_out == max(dim1, dim2) for dim1, dim2, dim_out in shapes)


@irdl_op_definition
class ClampOp(IRDLOperation):
    """
    Computes clamp(features, min, max)
    """

    name = "tosa.clamp"

    min_int = prop_def(IntegerAttr[I64])
    max_int = prop_def(IntegerAttr[I64])

    min_fp = prop_def(FloatAttr)
    max_fp = prop_def(FloatAttr)

    nan_mode = opt_prop_def(StringAttr)

    input = operand_def(TensorType)
    output = result_def(TensorType)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($output)"


@irdl_op_definition
class RescaleOp(IRDLOperation):
    """
    Tosa Rescale Operator
    """

    name = "tosa.rescale"

    input_zp = prop_def(IntegerAttr[I32])
    output_zp = prop_def(IntegerAttr[I32])
    multiplier = prop_def(DenseArrayBase)
    shift = prop_def(DenseArrayBase)
    scale32 = prop_def(BoolAttr)
    double_round = prop_def(BoolAttr)
    per_channel = prop_def(BoolAttr)

    input = operand_def(TensorType)
    output = result_def(TensorType)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($output)"


@irdl_op_definition
class AddOp(IRDLOperation):
    """
    Tosa elementwise add operation
    """

    name = "tosa.add"

    T: ClassVar = VarConstraint("T", AnyAttr())

    in1 = operand_def(TensorType.constr(T))
    in2 = operand_def(TensorType.constr(T))
    output = result_def(TensorType.constr(T))

    assembly_format = "$in1 `,` $in2 attr-dict `:` `(` type($in1) `,` type($in2) `)` `->` type($output)"

    def verify_(self) -> None:
        """
        Verify that the two input tensors are compatible, and that the result type can be constructed. For this,
        both tensors must have the same rank. They should either have the same number of elements per dim, or
        if there is only one element it can be broadcast implcitly.
        """
        t1 = self.in1.type
        t2 = self.in2.type
        t_out = self.output.type

        if not are_tosa_broadcastable(t1, t2, t_out):
            raise VerifyException(
                "'tosa.add' Operand and result tensor shapes are not compatible"
            )


TOSA = Dialect(
    "tosa",
    [
        ClampOp,
        RescaleOp,
        AddOp,
    ],
    [],
)
