"""
A dialect that represents at the highest level of abstraction the capabilities of the
[Snitch](https://github.com/pulp-platform/snitch_cluster) accelerator core, as used in
[Occamy](https://github.com/pulp-platform/occamy) and others.

The core aims to optimise for performance per watt, by replacing caches and branch
prediction logic with streaming registers and fixed-repetition loops. This dialect models
the streaming functionality of the Snitch core.

`snitch_stream.stride_pattern_type` represents a specification of the order in which
elements of a streamed region of memory will be read from or written to.

`snitch_stream.stride_pattern` creates a value storing the above specification.

`snitch_stream.streaming_region` encapsulates a region of code where the streams are
valid. According to the Snitch ABI, within this region, the registers `ft0` to `ftn`,
where `n` is the number of streaming registers, have a restricted functionality. If the
register is configured as a readable stream register, then it cannot be written to, and
if the register is configured as a writable stream register, then it cannot be read from.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class StridePattern(ParametrizedAttribute):
    """
    Attribute representing the order and offsets in which elements will be read from or
    written to a stream.

    ```
    // 2D access pattern
    #pat = #snitch_stream.stride_pattern<ub = [8, 16], strides = [128, 8]>
    // Corresponds to the following locations
    // for i in range(16):
    //   for j in range(8):
    //     yield i * 8 + j * 128
    // Note that the upper bounds and strides go from the innermost loop outwards
    ```
    """

    name = "snitch_stream.stride_pattern"

    ub: ParameterDef[ArrayAttr[IntAttr]]
    strides: ParameterDef[ArrayAttr[IntAttr]]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_identifier("ub")
            parser.parse_punctuation("=")
            ub = ArrayAttr(
                IntAttr(i)
                for i in parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            )
            parser.parse_punctuation(",")
            parser.parse_identifier("strides")
            parser.parse_punctuation("=")
            strides = ArrayAttr(
                IntAttr(i)
                for i in parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            )
            return (ub, strides)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("ub = [")
            printer.print_list(self.ub, lambda attr: printer.print(attr.data))
            printer.print_string("], strides = [")
            printer.print_list(self.strides, lambda attr: printer.print(attr.data))
            printer.print_string("]")

    def rank(self):
        return len(self.ub)

    def verify(self) -> None:
        if len(self.ub) != len(self.strides):
            raise VerifyException(
                f"Expect stride pattern upper bounds {self.ub} to be equal in length to strides {self.strides}"
            )


@irdl_attr_definition
class StridePatternType(Data[int], TypeAttribute):
    name = "snitch_stream.stride_pattern_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            return parser.parse_integer()

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))


@irdl_op_definition
class StreamingRegionOp(IRDLOperation):
    """
    An operation that creates streams from access patterns, which are only available to
    read from and write to within the body of the operation.

    According to the Snitch ABI, within this region, the registers `ft0` to `ftn`,
    where `n` is the number of streaming registers, have a restricted functionality. If the
    register is configured as a readable stream register, then it cannot be written to, and
    if the register is configured as a writable stream register, then it cannot be read from.
    """

    name = "snitch_stream.streaming_region"

    inputs = var_operand_def(riscv.IntRegisterType)
    """
    Pointers to memory buffers that will be streamed. The corresponding stride pattern
    defines the order in which the elements of the input buffers will be read.
    """
    outputs = var_operand_def(riscv.IntRegisterType)
    """
    Pointers to memory buffers that will be streamed. The corresponding stride pattern
    defines the order in which the elements of the input buffers will be written to.
    """
    stride_patterns = var_operand_def(StridePatternType)
    """
    Stride patterns that define the order of the input and output streams. If there is
    one stride pattern, and more inputs and outputs, the stride pattern is applied to all
    the streams.
    """

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset((NoTerminator(),))

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        stride_patterns: Sequence[SSAValue],
        body: Region,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs, stride_patterns],
            regions=[body],
        )


@irdl_op_definition
class StridePatternOp(IRDLOperation):
    """
    Specifies a stream access pattern reading from or writing to a pointer.
    `ub` specifies the upper bounds of the iteration variables.
    `strides` specifies the strides in bytes of the iteration variables.

    For example, to read sequentially the elements of a 2x3xf32 matrix in row-major order:
    `ub = [3, 2], strides = [4, 12]`

    The index for each iteration will be calculated like this:
    (0, 0) -> 0*12 + 0*4 = 0
    (0, 1) -> 0*12 + 1*4 = 4
    (0, 2) -> 0*12 + 2*4 = 8
    (1, 0) -> 1*12 + 0*4 = 12
    (1, 1) -> 1*12 + 1*4 = 16
    (1, 2) -> 1*12 + 2*4 = 18
    """

    name = "snitch_stream.stride_pattern"

    pattern = result_def(StridePatternType)
    ub = attr_def(ArrayAttr[IntAttr])
    strides = attr_def(ArrayAttr[IntAttr])
    dm = attr_def(IntAttr)

    def __init__(
        self,
        ub: ArrayAttr[IntAttr],
        strides: ArrayAttr[IntAttr],
        dm: IntAttr,
    ):
        rank = len(ub.data)
        assert rank == len(strides.data)
        super().__init__(
            result_types=[StridePatternType(rank)],
            attributes={
                "ub": ub,
                "strides": strides,
                "dm": dm,
            },
        )


SnitchStream = Dialect(
    "snitch_stream",
    [
        StreamingRegionOp,
        StridePatternOp,
    ],
    [
        StridePattern,
        StridePatternType,
    ],
)
