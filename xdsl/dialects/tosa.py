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
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


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

    in1 = operand_def(TensorType)
    in2 = operand_def(TensorType)
    output = result_def(TensorType)

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

        assert isa(t1, TensorType)
        assert isa(t2, TensorType)
        assert isa(t_out, TensorType)
        
        if not (t1.get_num_dims() == t2.get_num_dims() == t_out.get_num_dims()):
            raise VerifyException(
                "'tosa.add' rank mismatch between input and output tensors"
            )

        if not (
            t1.get_element_type() == t2.get_element_type() == t_out.get_element_type()
        ):
            raise VerifyException(
                "'tosa.add' element type mismatch between inputs and output tensors"
            )

        # tosa allows for implcit broadcasting (i.e. same dims but any '1' element dimension
        # can broadcast across that dimension) so we should check for same shapes, besides
        # '1's can appear anywhere
        s1, s2, s_out = t1.get_shape(), t2.get_shape(), t_out.get_shape()

        for dim_in1, dim_in2, dim_out in zip(s1, s2, s_out):
            # check that the shapes are compatible or have '1' to be broadcast
            if dim_in1 != dim_in2 and 1 not in (dim_in1, dim_in2):
                raise VerifyException(
                    f"'tosa.add' shapes mismatch along axis: {dim_in1} and {dim_in2}"
                )

            # check that t_out is made from only broadcasting the operand types
            if dim_out != max(dim_in1, dim_in2):
                raise VerifyException(
                    "'tosa.add' incompatible result type from operand types"
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
