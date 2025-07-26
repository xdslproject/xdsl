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
    traits_def,
)
from xdsl.traits import Commutative, Pure
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

    input1 = operand_def(TensorType.constr(T))
    input2 = operand_def(TensorType.constr(T))
    output = result_def(TensorType.constr(T))

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    traits = traits_def(
        Pure(),
        Commutative(),
    )

    def verify_(self) -> None:
        """
        Verify that the two input tensors are compatible, and that the result type can be constructed. For this,
        both tensors must have the same rank. They should either have the same number of elements per dim, or
        if there is only one element it can be broadcast implcitly.
        """
        t1 = self.input1.type
        t2 = self.input2.type
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
