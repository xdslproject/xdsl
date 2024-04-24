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
    T2 = Annotated[IntegerType, ConstraintVar("T")]

    frame = operand_def(TensorType[T])
    n_frame = operand_def(TensorType[T2])
    res = result_def(TensorType[T])

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
        pass


DSP = Dialect(
    "dsp",
    [
        STFT,
    ],
)
