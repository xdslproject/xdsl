from typing import Annotated

from xdsl.dialects.builtin import (
    AnyFloat,
    IntegerAttr,
    IntegerType,
    SSAValue,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class STFT(IRDLOperation):
    """
    The Short-Time Fourier Transform (STFT) is a technique used in digital signal processing to analyze the frequency content of a signal over time.
    It provides a time-frequency representation of a signal by computing the Fourier Transform over short, overlapping windows of the signal.
    This analysis is useful for tasks such as audio analysis, speech processing, and image processing.

    X[m,k]=∑ x[n]⋅w[n-m]⋅e^(-j*(2π/N)*nk)

    Where x[n] is the signal, w[n] is the window
    """

    name = "onnx.STFT"
    T = Annotated[AnyFloat, ConstraintVar("T")]

    frame = operand_def(TensorType[T])
    n_frame = operand_def(IntegerType)
    output = result_def(TensorType[T])

    frame_size = opt_attr_def(IntegerAttr, attr_name="frame_size")
    hop_size = opt_attr_def(IntegerAttr, attr_name="hop_size")

    assembly_format = (
        "`(` $operand`)` attr-dict `:` `(` type($operand) `)` `->` type($res)"
    )

    def __init__(
        self,
        frame: SSAValue,
        n_frame: SSAValue,
        frame_size: Attribute,
        hop_size: Attribute,
    ):
        super().__init__(
            attributes={
                "frame_size": frame_size,
                "hop_size": hop_size,
            },
            operands=[
                frame,
                n_frame,
            ],
            result_types=[frame.type],
        )

    def verify_(self) -> None:
        if (
            not isinstance(frame_type := self.frame.type, TensorType)
            or not isinstance(n_frame := self.n_frame.type, IntegerType)
            or not isinstance(output_type := self.output.type, TensorType)
        ):
            assert (
                False
            ), "dsp stft operation operands must be TensorType and IntegerType, the result must be of type TensorType"

        frame_shape = frame_type.get_shape()
        output_shape = output_type.get_shape()
        print(output_shape)
        print(n_frame)

        n_dimensions_frame = len(frame_shape)
        if n_dimensions_frame != 1:
            raise VerifyException(
                f"frame number of dimensions must be 1. Actual number of dimensions: {n_dimensions_frame}"
            )


DSP = Dialect(
    "dsp",
    [
        STFT,
    ],
)
