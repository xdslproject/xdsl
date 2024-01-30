from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, cast

from typing_extensions import Self

from xdsl.dialects import riscv, stream
from xdsl.dialects.builtin import IntAttr, UnrealizedConversionCastOp
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    RdRsImmIntegerOperation,
    RdRsRsOperation,
    Registers,
    RISCVInstruction,
    RISCVOp,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_assignment,
    print_assignment,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalisationPatternsTrait,
    HasParent,
    IsTerminator,
    Pure,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException

# region Snitch Extensions


class ScfgwOpHasCanonicalizationPatternsTrait(HasCanonicalisationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ScfgwOpUsingImmediate,
        )

        return (ScfgwOpUsingImmediate(),)


@irdl_op_definition
class ScfgwOp(RdRsRsOperation[IntRegisterType, IntRegisterType, IntRegisterType]):
    """
    Write the value in rs1 to the Snitch stream configuration
    location pointed by rs2 in the memory-mapped address space.
    Register rd is always fixed to zero.

    This is a RISC-V ISA extension, part of the `Xssr' extension.
    https://pulp-platform.github.io/snitch/rm/custom_instructions/
    """

    name = "riscv_snitch.scfgw"

    traits = frozenset((ScfgwOpHasCanonicalizationPatternsTrait(),))

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        # rd is always zero, so we omit it when printing assembly
        return self.rs1, self.rs2

    def verify_(self) -> None:
        if cast(IntRegisterType, self.rd.type) != Registers.ZERO:
            raise VerifyException(f"scfgw rd must be ZERO, got {self.rd.type}")


@irdl_op_definition
class ScfgwiOp(RdRsImmIntegerOperation):
    """
    Write the value in rs to the Snitch stream configuration location pointed by
    immediate value in the memory-mapped address space.

    This is a RISC-V ISA extension, part of the `Xssr' extension.
    https://pulp-platform.github.io/snitch/rm/custom_instructions/
    """

    name = "riscv_snitch.scfgwi"

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        # rd is always zero, so we omit it when printing assembly
        return self.rs1, self.immediate

    def verify_(self) -> None:
        if cast(IntRegisterType, self.rd.type) != Registers.ZERO:
            raise VerifyException(f"scfgwi rd must be ZERO, got {self.rd.type}")


@irdl_op_definition
class FrepYieldOp(AbstractYieldOperation[Attribute], RISCVOp):
    name = "riscv_snitch.frep_yield"

    traits = traits_def(
        lambda: frozenset([IsTerminator(), HasParent(FrepInner, FrepOuter)])
    )

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class ReadOp(IRDLOperation, RISCVOp):
    name = "riscv_snitch.read"

    T = Annotated[riscv.FloatRegisterType, ConstraintVar("T")]

    stream = operand_def(stream.ReadableStreamType[T])
    res = result_def(T)

    def __init__(self, stream_val: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream_type := stream_val.type, stream.ReadableStreamType)
            stream_type = cast(stream.ReadableStreamType[Attribute], stream_type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream_val], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> ReadOp:
        parser.parse_characters("from")
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved = parser.resolve_operand(
            unresolved, stream.ReadableStreamType(result_type)
        )
        return ReadOp(resolved, result_type)

    def print(self, printer: Printer):
        printer.print_string(" from ")
        printer.print(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class WriteOp(IRDLOperation, RISCVOp):
    name = "riscv_snitch.write"

    T = Annotated[riscv.FloatRegisterType, ConstraintVar("T")]

    value = operand_def(T)
    stream = operand_def(stream.WritableStreamType[T])

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])

    @classmethod
    def parse(cls, parser: Parser) -> WriteOp:
        unresolved_value = parser.parse_unresolved_operand()
        parser.parse_characters("to")
        unresolved_stream = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved_value = parser.resolve_operand(unresolved_value, result_type)
        resolved_stream = parser.resolve_operand(
            unresolved_stream, stream.WritableStreamType(result_type)
        )
        return WriteOp(resolved_value, resolved_stream)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(" to ")
        printer.print_ssa_value(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)

    def assembly_line(self) -> str | None:
        return None


ALLOWED_FREP_OP_TYPES = (
    FrepYieldOp,
    ReadOp,
    WriteOp,
    UnrealizedConversionCastOp,
)


class FRepOperation(IRDLOperation, RISCVInstruction):
    """
    From the Snitch paper: https://arxiv.org/abs/2002.10143

    The frep instruction marks the beginning of a floating-point kernel which should be
    repeated. It indicates how many subsequent instructions are stored in the sequence
    buffer, how often and how (operand staggering, repetition mode) each instruction is
    going to be repeated.
    """

    max_rep = operand_def(IntRegisterType)
    """Number of times to repeat the instructions."""
    body = region_def("single_block")
    """
    Instructions to repeat, containing maximum 15 instructions, with no side effects.
    """
    iter_args = var_operand_def(riscv.RISCVRegisterType)
    """
    Loop-carried variable initial values.
    """
    stagger_mask = attr_def(IntAttr)
    """
    4 bits for each operand (rs1 rs2 rs3 rd). If the bit is set, the corresponding operand
    is staggered.
    """
    stagger_count = attr_def(IntAttr)
    """
    3 bits, indicating for how many iterations the stagger should increment before it
    wraps again (up to 23 = 8).
    """
    res = var_result_def(riscv.RISCVRegisterType)
    """
    Loop-carried variable initial values.
    """

    traits = traits_def(
        lambda: frozenset((SingleBlockImplicitTerminator(FrepYieldOp),))
    )

    def __init__(
        self,
        max_rep: SSAValue | Operation,
        body: Sequence[Operation] | Sequence[Block] | Region,
        iter_args: Sequence[SSAValue | Operation] = (),
        stagger_mask: IntAttr | None = None,
        stagger_count: IntAttr | None = None,
    ):
        if stagger_mask is None:
            stagger_mask = IntAttr(0)
        if stagger_count is None:
            stagger_count = IntAttr(0)
        super().__init__(
            operands=(max_rep, iter_args),
            result_types=[[SSAValue.get(a).type for a in iter_args]],
            regions=(body,),
            attributes={
                "stagger_mask": stagger_mask,
                "stagger_count": stagger_count,
            },
        )

    @property
    def max_inst(self) -> int:
        """
        Number of instructions to be repeated.
        """
        return len([op for op in self.body.ops if isinstance(op, RISCVInstruction)])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (
            self.max_rep,
            self.max_inst,
            self.stagger_mask.data,
            self.stagger_count.data,
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        max_rep = parser.parse_operand()
        if parser.parse_optional_punctuation(","):
            stagger_mask = parser.parse_integer(False, False)
            parser.parse_punctuation(",")
            stagger_count = parser.parse_integer(False, False)
        else:
            stagger_mask = 0
            stagger_count = 0

        remaining_attributes = parser.parse_optional_attr_dict_with_keyword()

        # Parse iteration arguments
        pos = parser.pos
        unresolved_iter_args: list[Parser.UnresolvedArgument] = []
        iter_arg_unresolved_operands: list[UnresolvedOperand] = []
        iter_arg_types: list[Attribute] = []
        if parser.parse_optional_characters("iter_args"):
            for iter_arg, iter_arg_operand in parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, lambda: parse_assignment(parser)
            ):
                unresolved_iter_args.append(iter_arg)
                iter_arg_unresolved_operands.append(iter_arg_operand)
            parser.parse_characters("->")
            iter_arg_types = parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, parser.parse_attribute
            )

        iter_arg_operands = parser.resolve_operands(
            iter_arg_unresolved_operands, iter_arg_types, pos
        )

        # Set block argument types
        iter_args = [
            u_arg.resolve(t) for u_arg, t in zip(unresolved_iter_args, iter_arg_types)
        ]

        body = parser.parse_region(iter_args)

        frep = cls(
            max_rep,
            body,
            iter_arg_operands,
            IntAttr(stagger_mask),
            IntAttr(stagger_count),
        )
        if remaining_attributes is not None:
            frep.attributes |= remaining_attributes.data

        for trait in frep.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(frep, trait)

        return frep

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.max_rep)
        if self.stagger_count.data and self.stagger_mask.data:
            printer.print_string(", ")
            printer.print(self.stagger_count.data)
            printer.print_string(", ")
            printer.print(self.stagger_mask.data)

        printer.print_op_attributes(
            self.attributes, reserved_attr_names=("stagger_count", "stagger_mask")
        )
        printer.print_string(" ")

        block = self.body.block

        yield_op = block.last_op
        print_block_terminators = not isinstance(yield_op, FrepYieldOp) or bool(
            yield_op.operands
        )

        if iter_args := block.args:
            printer.print_string("iter_args(")
            printer.print_list(
                zip(iter_args, self.iter_args),
                lambda pair: print_assignment(printer, *pair),
            )
            printer.print_string(") -> (")
            printer.print_list((a.type for a in iter_args), printer.print_attribute)
            printer.print_string(") ")

        printer.print_region(
            self.body,
            print_entry_block_args=False,
            print_block_terminators=print_block_terminators,
        )

    def verify_(self) -> None:
        if self.stagger_count.data:
            raise VerifyException("Non-zero stagger count currently unsupported")
        if self.stagger_mask.data:
            raise VerifyException("Non-zero stagger mask currently unsupported")
        for instruction in self.body.ops:
            if not instruction.has_trait(Pure) and not isinstance(
                instruction, ALLOWED_FREP_OP_TYPES
            ):
                raise VerifyException(
                    "Frep operation body may not contain instructions "
                    f"with side-effects, found {instruction.name}"
                )
        if len(self.iter_args) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args)}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments."
            )
        for idx, (arg, block_arg) in enumerate(
            zip(self.iter_args, self.body.block.args)
        ):
            if block_arg.type != arg.type:
                raise VerifyException(
                    f"Block argument {idx} has wrong type, expected {arg.type}, "
                    f"got {block_arg.type}. Arguments after the "
                    f"induction variable must match the carried variables."
                )
        if len(self.body.ops) > 0 and isinstance(
            yieldop := self.body.block.last_op, FrepYieldOp
        ):
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The riscv_scf.frep must yield its carried variables."
                )
            for iter_arg, yield_arg in zip(self.iter_args, yieldop.arguments):
                if iter_arg.type != yield_arg.type:
                    raise VerifyException(
                        f"Expected {iter_arg.type}, got {yield_arg.type}. The "
                        f"riscv_snitch.frep's riscv_snitch.frep_yield must match carried"
                        f"variables types."
                    )


@irdl_op_definition
class FrepOuter(FRepOperation):
    """
    Repeats the instruction in the body as if the body were the body of a for loop, for
    example:

    ```
    # Repeat 4 times, stagger 1, period 2
    li a0, 4
    frep.o a0, 2, 1, 0b1010
    fadd.d fa0, ft0, ft2
    fmul.d fa0, ft3, fa0
    ```

    is equivalent to:
    ```
    fadd.d fa0, ft0, ft2
    fmul.d fa0, ft3, fa0
    fadd.d fa1, ft0, ft3
    fmul.d fa1, ft3, fa1
    fadd.d fa0, ft0, ft2
    fmul.d fa0, ft3, fa0
    fadd.d fa1, ft0, ft3
    fmul.d fa1, ft3, fa1
    ```
    """

    name = "riscv_snitch.frep_outer"

    def assembly_instruction_name(self) -> str:
        return "frep.o"


@irdl_op_definition
class FrepInner(FRepOperation):
    """
    Repeats the instruction in the body, as if each were in its own body of a for loop,
    for example:

    ```
    # Repeat three times, stagger 2, period 2
    li a0, 3
    frep.i a0, 2, 2, 0b0100
    fadd.d fa0, ft0, ft2
    fmul.d fa0, ft3, fa0
    ```

    is equivalent to:
    ```
    fadd.d fa0, ft0, ft2
    fadd.d fa0, ft1, ft3
    fadd.d fa0, ft2, ft3
    fmul.d fa0, ft3, fa0
    fmul.d fa0, ft4, fa0
    fmul.d fa0, ft5, fa0
    ```
    """

    name = "riscv_snitch.frep_inner"

    def assembly_instruction_name(self) -> str:
        return "frep.i"


@irdl_op_definition
class GetStreamOp(IRDLOperation, RISCVOp):
    name = "riscv_snitch.get_stream"

    stream = result_def(stream.StreamType[riscv.FloatRegisterType])

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> GetStreamOp:
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        return GetStreamOp(result_type)

    def print(self, printer: Printer):
        printer.print_string(" : ")
        printer.print_attribute(self.stream.type)

    def assembly_line(self) -> str | None:
        return None


# endregion

RISCV_Snitch = Dialect(
    "riscv_snitch",
    [
        ScfgwOp,
        ScfgwiOp,
        FrepOuter,
        FrepInner,
        FrepYieldOp,
        ReadOp,
        WriteOp,
        GetStreamOp,
    ],
    [],
)
