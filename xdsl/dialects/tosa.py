from xdsl.dialects.builtin import (
    I32,
    I64,
    AnyFloat,
    BoolAttr,
    DenseArrayBase,
    FloatAttr,
    IntegerAttr,
    TensorType,
)
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)


@irdl_op_definition
class ClampOp(IRDLOperation):
    """
    Computes clamp(features, min, max)
    """

    name = "tosa.clamp"

    min_int = opt_attr_def(IntegerAttr[I64])
    max_int = opt_attr_def(IntegerAttr[I64])

    min_fp = opt_attr_def(FloatAttr[AnyFloat])
    max_fp = opt_attr_def(FloatAttr[AnyFloat])

    input = operand_def(TensorType)
    output = result_def(TensorType)

    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($output)"


@irdl_op_definition
class RescaleOp(IRDLOperation):
    """
    Tosa Rescale Operator
    """

    name = "tosa.rescale"

    input_zp = attr_def(IntegerAttr[I32])
    output_zp = attr_def(IntegerAttr[I32])
    multiplier = attr_def(DenseArrayBase)
    shift = attr_def(DenseArrayBase)
    scale32 = attr_def(BoolAttr)
    double_round = attr_def(BoolAttr)
    per_channel = attr_def(BoolAttr)

    input = operand_def(TensorType)
    output = result_def(TensorType)

    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($output)"


TOSA = Dialect("tosa", [ClampOp, RescaleOp], [])
