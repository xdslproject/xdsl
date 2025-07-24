from typing import ClassVar

from xdsl.dialects.builtin import (
    I32,
    I64,
    AnyAttr,
    BoolAttr,
    DenseArrayBase,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    TensorType,
)
from xdsl.ir import Dialect
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
from xdsl.utils.type import are_tosa_broadcastable


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
