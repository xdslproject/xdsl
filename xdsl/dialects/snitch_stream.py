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
from typing import cast

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
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
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

    ub: ArrayAttr[IntAttr]
    strides: ArrayAttr[IntAttr]
    repeat: IntAttr
    """
    Number of times an element will be repeated when loaded, default is 1.
    """

    def __init__(
        self,
        ub: ArrayAttr[IntAttr],
        strides: ArrayAttr[IntAttr],
        repeat: IntAttr = IntAttr(1),
    ):
        super().__init__(ub, strides, repeat)

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
            if parser.parse_optional_punctuation(","):
                parser.parse_identifier("repeat")
                parser.parse_punctuation("=")
                repeat = parser.parse_integer(allow_boolean=False, allow_negative=False)
            else:
                repeat = 1
            return (ub, strides, IntAttr(repeat))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("ub = ")
            with printer.in_square_brackets():
                printer.print_list(self.ub, lambda attr: printer.print_int(attr.data))
            printer.print_string(", strides = ")
            with printer.in_square_brackets():
                printer.print_list(
                    self.strides, lambda attr: printer.print_int(attr.data)
                )
            if self.repeat.data != 1:
                printer.print_string(", repeat = ")
                printer.print_int(self.repeat.data)

    @staticmethod
    def from_bounds_and_strides(
        ub: Sequence[int], strides: Sequence[int], repeat: int = 1
    ) -> StridePattern:
        return StridePattern(
            ArrayAttr(IntAttr(i) for i in ub),
            ArrayAttr(IntAttr(i) for i in strides),
            IntAttr(repeat),
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
            offset = sum(
                index * stride.data
                for (index, stride) in zip(indices, self.strides.data)
            )
            for _ in range(self.repeat.data):
                yield offset

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

        if not tuples:
            # All bounds are 1
            return StridePattern.from_bounds_and_strides(
                (1,), (self.strides.data[-1].data,)
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

        if s[-1] == 0:
            repeat = ub[-1] * self.repeat.data
            ub = ub[:-1]
            s = s[:-1]
        else:
            repeat = self.repeat.data

        return StridePattern.from_bounds_and_strides(ub, s, repeat)


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

    traits = traits_def(NoTerminator())

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

    def print(self, printer: Printer):
        with printer.indented():
            printer.print_string(" {")
            if self.stride_patterns.data:
                printer.print_string("\npatterns = [")
                with printer.indented():
                    if self.stride_patterns.data:
                        printer.print_string("\n")
                        printer.print_list(
                            self.stride_patterns.data,
                            printer.print_attribute,
                            delimiter=",\n",
                        )
                printer.print_string("\n]")
            else:
                printer.print_string("\npatterns = []")
        printer.print_string("\n}")

        if self.inputs:
            printer.print_string(" ins(")
            printer.print_list(self.inputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.inputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.outputs:
            printer.print_string(" outs(")
            printer.print_list(self.outputs, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.outputs.types, printer.print_attribute)
            printer.print_string(")")

        if self.attributes:
            printer.print_string(" attributes = ")
            printer.print_op_attributes(self.attributes)

        printer.print_string(" ")
        printer.print_region(self.body)

    @classmethod
    def parse(cls, parser: Parser) -> StreamingRegionOp:
        parser.parse_punctuation("{")
        parser.parse_identifier("stride_patterns")
        parser.parse_punctuation("=")

        patterns = parser.parse_attribute()
        if not isinstance(patterns, ArrayAttr):
            parser.raise_error(f"Expected ArrayAttr {patterns}")
        patterns = cast(ArrayAttr[Attribute], patterns)
        for pattern in patterns:
            if not isinstance(pattern, StridePattern):
                parser.raise_error(f"Expected StridePattern {pattern}")
        patterns = cast(ArrayAttr[StridePattern], patterns)

        parser.parse_punctuation("}")

        pos = parser.pos
        if parser.parse_optional_characters("ins"):
            parser.parse_punctuation("(")
            unresolved_ins = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            ins_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            ins = parser.resolve_operands(unresolved_ins, ins_types, pos)
        else:
            ins = ()

        pos = parser.pos
        if parser.parse_optional_characters("outs"):
            parser.parse_punctuation("(")
            unresolved_outs = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            outs_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            outs = parser.resolve_operands(unresolved_outs, outs_types, pos)
        else:
            outs = ()

        if parser.parse_optional_keyword("attributes"):
            parser.parse_punctuation("=")
            extra_attrs = parser.expect(
                parser.parse_optional_attr_dict, "expect extra attributes"
            )
        else:
            extra_attrs = {}

        body = parser.parse_region()

        generic = cls(
            ins,
            outs,
            patterns,
            body,
        )
        generic.attributes |= extra_attrs

        return generic


SnitchStream = Dialect(
    "snitch_stream",
    [
        StreamingRegionOp,
    ],
    [
        StridePattern,
    ],
)
