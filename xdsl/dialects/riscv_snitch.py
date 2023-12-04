from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    IntAttr,
)
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    RdRsImmIntegerOperation,
    RdRsRsOperation,
    Registers,
    RISCVInstruction,
    RISCVOp,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    region_def,
    traits_def,
)
from xdsl.parser import Parser
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

    traits = traits_def(
        lambda: frozenset((SingleBlockImplicitTerminator(FrepYieldOp),))
    )

    def __init__(
        self,
        max_rep: SSAValue | Operation,
        body: Sequence[Operation] | Sequence[Block] | Region,
        stagger_mask: IntAttr,
        stagger_count: IntAttr,
    ):
        super().__init__(
            operands=(max_rep,),
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

        body = parser.parse_region()

        frep = cls(max_rep, body, IntAttr(stagger_mask), IntAttr(stagger_count))
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

        yield_op = self.body.block.last_op
        print_block_terminators = not isinstance(yield_op, FrepYieldOp) or bool(
            yield_op.operands
        )

        printer.print_region(self.body, print_block_terminators=print_block_terminators)

    def verify_(self) -> None:
        if self.stagger_count.data:
            raise VerifyException("Non-zero stagger count currently unsupported")
        if self.stagger_mask.data:
            raise VerifyException("Non-zero stagger mask currently unsupported")
        for instruction in self.body.ops:
            if not instruction.has_trait(Pure) and not isinstance(
                instruction, FrepYieldOp
            ):
                raise VerifyException(
                    "Frep operation body may not contain instructions "
                    f"with side-effects, found {instruction.name}"
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
class FrepYieldOp(AbstractYieldOperation[Attribute], RISCVOp):
    name = "riscv_snitch.frep_yield"

    traits = traits_def(
        lambda: frozenset([IsTerminator(), HasParent(FrepInner, FrepOuter)])
    )

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
    ],
    [],
)
