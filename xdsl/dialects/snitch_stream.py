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

from collections.abc import Iterator, Sequence
from itertools import product

from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    region_def,
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
    #pat = #snitch_stream.stride_pattern<ub = [16, 8], strides = [8, 128]>
    // Corresponds to the following locations
    // for i in range(16):
    //   for j in range(8):
    //     yield i * 8 + j * 128
    // Note that the upper bounds and strides go from the outermost loop inwards
    ```
    """

    name = "snitch_stream.stride_pattern"

    ub: ParameterDef[ArrayAttr[IntAttr]]
    strides: ParameterDef[ArrayAttr[IntAttr]]

    def __init__(
        self,
        ub: ParameterDef[ArrayAttr[IntAttr]],
        strides: ParameterDef[ArrayAttr[IntAttr]],
    ):
        super().__init__((ub, strides))

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

    @staticmethod
    def from_bounds_and_strides(
        ub: Sequence[int], strides: Sequence[int]
    ) -> StridePattern:
        return StridePattern(
            ArrayAttr(IntAttr(i) for i in ub),
            ArrayAttr(IntAttr(i) for i in strides),
        )

    def rank(self):
        return len(self.ub)

    def verify(self) -> None:
        if len(self.ub) != len(self.strides):
            raise VerifyException(
                f"Expect stride pattern upper bounds {self.ub} to be equal in length to strides {self.strides}"
            )

    def offset_iter(self) -> Iterator[int]:
        for indices in product(*(range(bound.data) for bound in self.ub.data)):
            indices: tuple[int, ...] = indices
            yield sum(
                index * stride.data
                for (index, stride) in zip(indices, self.strides.data)
            )

    def offsets(self) -> tuple[int, ...]:
        return tuple(self.offset_iter())

    def simplified(self) -> StridePattern:
        """
        Return a stride pattern that specifies the same iteration space, but with folded
        perfectly nested outermost loops.

        e.g.

        ```
        stride_pattern<ub = [2, 3, 4], strides = [12, 4, 1]>
        ->
        stride_pattern<ub = [24], strides = [1]
        ```
        """
        if len(self.ub) < 2:
            return self

        tuples = tuple(
            (bound.data, stride.data)
            for bound, stride in zip(self.ub.data, self.strides.data)
            # Exclude single iteration bounds
            if bound.data != 1
        )

        # Outermost bound and stride
        ub0, s0 = tuples[0]

        # Start with the second outermost loop bounds
        second_outermost_dim = 1
        while second_outermost_dim < len(tuples):
            # Next bound and stride to fold into outermost
            ubd, sd = tuples[second_outermost_dim]
            if s0 == ubd * sd:
                # The second outermost loop is perfectly nested in outermost
                ub0 = ub0 * ubd
                s0 = sd
                # Decrement the index into tuples for what the new second outermost loop
                # bound is
                second_outermost_dim += 1
            else:
                # The second outermost loop does not match, do not try to further simplify
                break

        # ub and s include the new outermost bound and stride,
        # followed by all the tuples up to and including the second outermost dim
        ub = (ub0, *(bound for bound, _ in tuples[second_outermost_dim:]))
        s = (s0, *(stride for _, stride in tuples[second_outermost_dim:]))

        return StridePattern.from_bounds_and_strides(ub, s)


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
    stride_patterns = prop_def(ArrayAttr[StridePattern])
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
        stride_patterns: ArrayAttr[StridePattern],
        body: Region,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={
                "stride_patterns": stride_patterns,
            },
        )


SnitchStream = Dialect(
    "snitch_stream",
    [
        StreamingRegionOp,
    ],
    [
        StridePattern,
    ],
)
