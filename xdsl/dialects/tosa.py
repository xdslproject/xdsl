from xdsl.dialects.builtin import (
    I32,
    I64,
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
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
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


TOSA = Dialect(
    "tosa",
    [
        ClampOp,
        RescaleOp,
    ],
    [],
)
