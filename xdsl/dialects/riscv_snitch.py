from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, Literal, TypeAlias, cast

from typing_extensions import Self

from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.backend.register_allocator import BlockAllocator
from xdsl.backend.riscv.traits import StaticInsnRepresentation
from xdsl.dialects import riscv, snitch
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    FastMathFlagsAttr,
    FloatRegisterType,
    IntRegisterType,
    RdRsRsOperation,
    RISCVAsmOperation,
    RISCVCustomFormatOperation,
    RISCVInstruction,
    RISCVRegisterType,
    RsRsIntegerOperation,
    SImm12Attr,
    UImm5Attr,
    parse_immediate_value,
    print_immediate_value,
    si12,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_assignment,
    print_assignment,
)
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    BaseAttr,
    VarConstraint,
    attr_def,
    base,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_attr_def,
    prop_def,
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
    HasCanonicalizationPatternsTrait,
    HasParent,
    IsTerminator,
    Pure,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException

# region Snitch Extensions


class ScfgwOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ScfgwOpUsingImmediate,
        )

        return (ScfgwOpUsingImmediate(),)


@irdl_op_definition
class ScfgwOp(RsRsIntegerOperation):
    """
    Write the value in rs1 to the Snitch stream configuration
    location pointed by rs2 in the memory-mapped address space.

    This is a RISC-V ISA extension, part of the `Xssr' extension.

    See external [documentation](https://pulp-platform.github.io/snitch/rm/custom_instructions/).
    """

    name = "riscv_snitch.scfgw"

    traits = traits_def(ScfgwOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class ScfgwiOp(RISCVCustomFormatOperation, RISCVInstruction):
    """
    Write the value in rs to the Snitch stream configuration location pointed by
    immediate value in the memory-mapped address space.

    This is a RISC-V ISA extension, part of the `Xssr' extension.

    See external [documentation](https://pulp-platform.github.io/snitch/rm/custom_instructions/).
    """

    name = "riscv_snitch.scfgwi"

    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(SImm12Attr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | SImm12Attr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, si12)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, si12)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> set[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class FrepYieldOp(AbstractYieldOperation[Attribute], RISCVAsmOperation):
    name = "riscv_snitch.frep_yield"

    traits = lazy_traits_def(
        lambda: (IsTerminator(), HasParent(FrepInnerOp, FrepOuterOp))
    )

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class ReadOp(RISCVAsmOperation):
    name = "riscv_snitch.read"

    T: ClassVar = VarConstraint("T", AnyAttr())

    stream = operand_def(snitch.ReadableStreamType.constr(T))
    res = result_def(T)

    assembly_format = "`from` $stream attr-dict `:` type($res)"

    def __init__(self, stream_val: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream_type := stream_val.type, snitch.ReadableStreamType)
            stream_type = cast(snitch.ReadableStreamType[Attribute], stream_type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream_val], result_types=[result_type])

    def assembly_line(self) -> str | None:
        return None

    def iter_used_registers(self):
        # When streaming, FT0, FT1, and FT2 cannot be used as general-purpose float
        # registers
        yield riscv.Registers.FT0
        yield riscv.Registers.FT1
        yield riscv.Registers.FT2


@irdl_op_definition
class WriteOp(RISCVAsmOperation):
    name = "riscv_snitch.write"

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    stream = operand_def(snitch.WritableStreamType.constr(T))

    assembly_format = "$value `to` $stream attr-dict `:` type($value)"

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])

    def assembly_line(self) -> str | None:
        return None

    def iter_used_registers(self):
        # When streaming, FT0, FT1, and FT2 cannot be used as general-purpose float
        # registers
        yield riscv.Registers.FT0
        yield riscv.Registers.FT1
        yield riscv.Registers.FT2


ALLOWED_FREP_OP_TYPES = (
    FrepYieldOp,
    ReadOp,
    WriteOp,
    UnrealizedConversionCastOp,
)

I3: TypeAlias = IntegerType[Literal[3]]
I4: TypeAlias = IntegerType[Literal[4]]
i3 = I3(3)
i4 = I4(4)


class FRepOperation(RISCVInstruction):
    """
    The frep instruction marks the beginning of a floating-point kernel which should be
    repeated. It indicates how many subsequent instructions are stored in the sequence
    buffer, how often and how (operand staggering, repetition mode) each instruction is
    going to be repeated.

    Snitch paper: See external [documentation](https://arxiv.org/abs/2002.10143).
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
    stagger_mask = attr_def(
        IntegerAttr[I4],
        default_value=IntegerAttr(0, i4),
    )
    """
    4 bits for each operand (rs1 rs2 rs3 rd). If the bit is set, the corresponding operand
    is staggered.
    """
    stagger_count = attr_def(
        IntegerAttr[I3],
        default_value=IntegerAttr(0, i3),
    )
    """
    3 bits, indicating for how many iterations the stagger should increment before it
    wraps again (up to 23 = 8).
    """
    res = var_result_def(riscv.RISCVRegisterType)
    """
    Loop-carried variable initial values.
    """

    traits = lazy_traits_def(lambda: (SingleBlockImplicitTerminator(FrepYieldOp),))

    def __init__(
        self,
        max_rep: SSAValue | Operation,
        body: Sequence[Operation] | Sequence[Block] | Region,
        iter_args: Sequence[SSAValue | Operation] = (),
        stagger_mask: IntegerAttr[I4] | None = None,
        stagger_count: IntegerAttr[I3] | None = None,
    ):
        if stagger_mask is None:
            stagger_mask = IntegerAttr(0, i4)
        if stagger_count is None:
            stagger_count = IntegerAttr(0, i3)
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
            str(self.max_inst),
            self.stagger_mask,
            self.stagger_count,
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
            IntegerAttr(stagger_mask, i4),
            IntegerAttr(stagger_count, i3),
        )
        if remaining_attributes is not None:
            frep.attributes |= remaining_attributes.data

        for trait in frep.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(frep, trait)

        return frep

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.max_rep)
        if self.stagger_count.value.data and self.stagger_mask.value.data:
            printer.print_string(", ")
            printer.print_int(self.stagger_count.value.data)
            printer.print_string(", ")
            printer.print_int(self.stagger_mask.value.data)

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
        if self.stagger_count.value.data:
            raise VerifyException("Non-zero stagger count currently unsupported")
        if self.stagger_mask.value.data:
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

    def allocate_registers(self, allocator: BlockAllocator) -> None:
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = allocator.live_ins_per_block[self.body.block]
        for live_in in live_ins:
            allocator.allocate_value(live_in)

        yield_op = self.body.block.last_op
        assert yield_op is not None, (
            "last op of riscv_snitch.frep_outer and riscv_snitch.frep_inner is guaranteed"
            " to be riscv_scf.Yield"
        )
        block_args = self.body.block.args

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args, self.iter_args, yield_op.operands, self.results
        ):
            allocator.allocate_values_same_reg(
                (block_arg, operand, yield_operand, op_result)
            )

        allocator.allocate_value(self.max_rep)

        # Reserve the loop carried variables for allocation within the body
        regs = self.iter_args.types
        assert all(isinstance(reg, RISCVRegisterType) for reg in regs)
        regs = cast(tuple[RISCVRegisterType, ...], regs)
        with allocator.available_registers.reserve_registers(regs):
            allocator.allocate_block(self.body.block)


@irdl_op_definition
class FrepOuterOp(FRepOperation):
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
class FrepInnerOp(FRepOperation):
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
class GetStreamOp(RISCVAsmOperation):
    name = "riscv_snitch.get_stream"

    stream = result_def(
        snitch.ReadableStreamType.constr(BaseAttr(riscv.FloatRegisterType))
        | snitch.WritableStreamType.constr(BaseAttr(riscv.FloatRegisterType))
    )

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

# region XDMA extensions
# Documentation for these operations:
# https://pulp-platform.github.io/snitch_cluster/rm/custom_instructions.html


@irdl_op_definition
class DMSourceOp(RISCVCustomFormatOperation, RISCVInstruction):
    name = "riscv_snitch.dmsrc"

    ptrlo = operand_def(riscv.IntRegisterType)
    ptrhi = operand_def(riscv.IntRegisterType)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 0, x0, {0}, {1}")
    )

    def __init__(self, ptrlo: SSAValue | Operation, ptrhi: SSAValue | Operation):
        super().__init__(operands=[ptrlo, ptrhi])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.ptrlo, self.ptrhi


@irdl_op_definition
class DMDestinationOp(RISCVCustomFormatOperation, RISCVInstruction):
    name = "riscv_snitch.dmdst"

    ptrlo = operand_def(riscv.IntRegisterType)
    ptrhi = operand_def(riscv.IntRegisterType)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 1, x0, {0}, {1}")
    )

    def __init__(self, ptrlo: SSAValue | Operation, ptrhi: SSAValue | Operation):
        super().__init__(operands=[ptrlo, ptrhi])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.ptrlo, self.ptrhi


@irdl_op_definition
class DMStrideOp(RISCVCustomFormatOperation, RISCVInstruction):
    name = "riscv_snitch.dmstr"

    srcstrd = operand_def(riscv.IntRegisterType)
    dststrd = operand_def(riscv.IntRegisterType)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 6, x0, {0}, {1}")
    )

    def __init__(self, srcstrd: SSAValue | Operation, dststrd: SSAValue | Operation):
        super().__init__(operands=[srcstrd, dststrd])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.srcstrd, self.dststrd


@irdl_op_definition
class DMRepOp(RISCVCustomFormatOperation, RISCVInstruction):
    name = "riscv_snitch.dmrep"

    reps = operand_def(riscv.IntRegisterType)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 7, x0, {0}, x0")
    )

    def __init__(self, reps: SSAValue | Operation):
        super().__init__(operands=[reps])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.reps,)


@irdl_op_definition
class DMCopyOp(RISCVCustomFormatOperation, RISCVInstruction):
    name = "riscv_snitch.dmcpy"

    dest = result_def(riscv.IntRegisterType)
    size = operand_def(riscv.IntRegisterType)
    config = operand_def(riscv.IntRegisterType)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 3, {0}, {1}, {2}")
    )

    def __init__(
        self,
        size: SSAValue | Operation,
        config: SSAValue | Operation,
        result_type: IntRegisterType = riscv.Registers.UNALLOCATED_INT,
    ):
        super().__init__(operands=[size, config], result_types=[result_type])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.dest, self.size, self.config


@irdl_op_definition
class DMStatOp(RISCVCustomFormatOperation, RISCVInstruction):
    name = "riscv_snitch.dmstat"

    dest = result_def(riscv.IntRegisterType)
    status = operand_def(riscv.IntRegisterType)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 5, {0}, {1}, {2}")
    )

    def __init__(
        self,
        status: SSAValue | Operation,
        result_type: IntRegisterType = riscv.Registers.UNALLOCATED_INT,
    ):
        super().__init__(operands=[status], result_types=[result_type])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.dest, self.status


@irdl_op_definition
class DMCopyImmOp(RISCVInstruction):
    name = "riscv_snitch.dmcpyi"

    dest = result_def(riscv.IntRegisterType)
    size = operand_def(riscv.IntRegisterType)
    config = prop_def(UImm5Attr)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 2, {0}, {1}, {2}")
    )

    def __init__(
        self,
        size: SSAValue | Operation,
        config: int | UImm5Attr,
        result_type: IntRegisterType = riscv.Registers.UNALLOCATED_INT,
    ):
        if isinstance(config, int):
            config = IntegerAttr(config, IntegerType(5, signedness=Signedness.UNSIGNED))
        super().__init__(
            operands=[size],
            properties={"config": config},
            result_types=[result_type],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.dest, self.size, self.config

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.size)
        printer.print_string(", ")
        self.config.print_without_type(printer)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)
        printer.print_string(" : ")
        printer.print_operation_type(self)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        size = parser.parse_operand()
        parser.parse_punctuation(",")
        config = parser.parse_integer()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        signature = parser.parse_function_type()
        result_type, *_ = signature.outputs
        op = cls(size, config, cast(IntRegisterType, result_type))
        if attrs:
            op.attributes.update(attrs)
        return op


@irdl_op_definition
class DMStatImmOp(RISCVInstruction):
    name = "riscv_snitch.dmstati"

    dest = result_def(riscv.IntRegisterType)
    status = prop_def(UImm5Attr)

    traits = traits_def(
        StaticInsnRepresentation(insn=".insn r 0x2b, 0, 4, {0}, {1}, {2}")
    )

    def __init__(
        self,
        status: int | UImm5Attr,
        result_type: IntRegisterType = riscv.Registers.UNALLOCATED_INT,
    ):
        if isinstance(status, int):
            status = IntegerAttr(status, IntegerType(5, signedness=Signedness.UNSIGNED))
        super().__init__(
            properties={"status": status},
            result_types=[result_type],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.dest, self.status

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        self.status.print_without_type(printer)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)
        printer.print_string(" : ")
        printer.print_operation_type(self)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        status = parser.parse_integer()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        signature = parser.parse_function_type()
        result_type, *_ = signature.outputs
        op = cls(status, cast(IntRegisterType, result_type))
        if attrs:
            op.attributes.update(attrs)
        return op


# endregion

# region Snitch Packed SIMD Extension

# Operations that map directly to the packed SIMD ISA provided by Snitch FPU.
# The implemented ISA is *almost* the one specified here:
# * https://iis-git.ee.ethz.ch/smach/smallFloat-spec/-/blob/master/smallFloat_isa.pdf
# Beware of main undocumented differences from the spec:
# * Additional reductions (e.g.: vfsum.*)
# * Missing reductions (e.g.: vfdotp.*)
# * Control of alternative FP formats (e.g.: IEEE fp16 vs BF16) delegated to the
#   RISC-V float CSR instead of being part of the encoding


@irdl_op_definition
class VFCpkASSOp(
    RdRsRsOperation[FloatRegisterType, FloatRegisterType, FloatRegisterType]
):
    """
    Packs two scalar f32 values from rs1 and rs2 and packs the result as two adjacent
    entries into the vectorial 2xf32 rd operand, such as:

    ```C
    f[rd][lo] = f[rs1]
    f[rd][hi] = f[rs2]
    ```
    """

    name = "riscv_snitch.vfcpka.s.s"

    traits = traits_def(Pure())


@irdl_op_definition
class VFMulSOp(riscv.RdRsRsFloatOperationWithFastMath):
    """
    Performs vectorial multiplication of corresponding f32 values from
    rs1 and rs2 and stores the results in the corresponding f32 lanes
    into the vectorial 2xf32 rd operand, such as:

    ```C
    f[rd][lo] = f[rs1][lo] * f[rs2][lo]
    f[rd][hi] = f[rs1][hi] * f[rs2][hi]
    ```
    """

    name = "riscv_snitch.vfmul.s"

    traits = traits_def(Pure())


@irdl_op_definition
class VFAddSOp(riscv.RdRsRsFloatOperationWithFastMath):
    """
    Performs vectorial addition of corresponding f32 values from
    rs1 and rs2 and stores the results in the corresponding f32 lanes
    into the vectorial 2xf32 rd operand, such as:

    ```C
    f[rd][lo] = f[rs1][lo] + f[rs2][lo]
    f[rd][hi] = f[rs1][hi] + f[rs2][hi]
    ```
    """

    name = "riscv_snitch.vfadd.s"

    traits = traits_def(Pure())


@irdl_op_definition
class VFAddHOp(riscv.RdRsRsFloatOperationWithFastMath):
    """
    Performs vectorial addition of corresponding f16 values from
    rs1 and rs2 and stores the results in the corresponding f16 lanes
    into the vectorial 4xf16 rd operand, such as:

    ```C
    f[rd][0] = f[rs1][0] + f[rs2][0]
    f[rd][1] = f[rs1][1] + f[rs2][1]
    f[rd][2] = f[rs1][2] + f[rs2][2]
    f[rd][3] = f[rs1][3] + f[rs2][3]
    ```
    """

    name = "riscv_snitch.vfadd.h"

    traits = traits_def(Pure())


@irdl_op_definition
class VFMaxSOp(riscv.RdRsRsFloatOperationWithFastMath):
    """
    Performs vectorial maximum of corresponding f32 values from
    rs1 and rs2 and stores the results in the corresponding f32 lanes
    into the vectorial 2xf32 rd operand, such as:

    ```C
    f[rd][lo] = max(f[rs1][lo], f[rs2][lo])
    f[rd][hi] = max(f[rs1][hi], f[rs2][hi])
    ```
    """

    name = "riscv_snitch.vfmax.s"

    traits = traits_def(Pure())


class RdRsRsAccumulatingFloatOperationWithFastMath(
    RISCVCustomFormatOperation, RISCVInstruction, ABC
):
    """
    A base class for RISC-V operations that have one destination floating-point register,
    that also acts as a source register, and two source floating-point registers and can
    be annotated with fastmath flags.
    """

    SAME_FLOAT_REGISTER_TYPE: ClassVar = VarConstraint(
        "SAME_FLOAT_REGISTER_TYPE", base(FloatRegisterType)
    )

    rd_out = result_def(SAME_FLOAT_REGISTER_TYPE)
    rd_in = operand_def(SAME_FLOAT_REGISTER_TYPE)
    rs1 = operand_def(FloatRegisterType)
    rs2 = operand_def(FloatRegisterType)

    fastmath = opt_attr_def(FastMathFlagsAttr)

    def __init__(
        self,
        rd: Operation | SSAValue,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        fastmath: FastMathFlagsAttr | None = None,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(rd, Operation):
            rd = SSAValue.get(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rd, rs1, rs2],
            attributes={
                "fastmath": fastmath,
                "comment": comment,
            },
            result_types=[rd.type],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd_in, self.rs1, self.rs2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        flags = FastMathFlagsAttr("none")
        if parser.parse_optional_keyword("fastmath") is not None:
            flags = FastMathFlagsAttr(FastMathFlagsAttr.parse_parameter(parser))
        attributes["fastmath"] = flags
        return attributes

    def custom_print_attributes(self, printer: Printer) -> set[str]:
        if self.fastmath is not None and self.fastmath != FastMathFlagsAttr("none"):
            printer.print_string(" fastmath")
            self.fastmath.print_parameter(printer)
        return {"fastmath"}

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.rs1, self.rs2), (), ((self.rd_in, self.rd_out),)
        )


class RdRsAccumulatingFloatOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one destination floating-point register,
    that also acts as a source register, and a source floating-point register.
    """

    SAME_FLOAT_REGISTER_TYPE: ClassVar = VarConstraint(
        "SAME_FLOAT_REGISTER_TYPE", base(FloatRegisterType)
    )

    rd_out = result_def(SAME_FLOAT_REGISTER_TYPE)
    rd_in = operand_def(SAME_FLOAT_REGISTER_TYPE)
    rs = operand_def(FloatRegisterType)

    def __init__(
        self,
        rd: Operation | SSAValue,
        rs: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(rd, Operation):
            rd = SSAValue.get(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rd, rs],
            attributes={
                "comment": comment,
            },
            result_types=[rd.type],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd_in, self.rs

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints((self.rs,), (), ((self.rd_in, self.rd_out),))


@irdl_op_definition
class VFMacSOp(RdRsRsAccumulatingFloatOperationWithFastMath):
    """
    Performs vectorial multiplication of corresponding f32 values from
    rs1 and rs2 and accumulates the results in the corresponding f32 lanes
    into the vectorial 2xf32 rd operand, such as:

    ```C
    f[rd][lo] = f[rs1][lo] * f[rs2][lo] + f[rd][lo]
    f[rd][hi] = f[rs1][hi] * f[rs2][hi] + f[rd][hi]
    ```
    """

    name = "riscv_snitch.vfmac.s"

    traits = traits_def(Pure())


@irdl_op_definition
class VFSumSOp(RdRsAccumulatingFloatOperation):
    """
    Performs sum of f32 values from rs and accumulates the result in the lower f32 value
    of the rd operand:

    ```C
    f[rd][lo] = f[rs][hi] + f[rs][lo] + f[rd][lo]
    ```
    """

    name = "riscv_snitch.vfsum.s"

    traits = traits_def(Pure())


# endregion

RISCV_Snitch = Dialect(
    "riscv_snitch",
    [
        ScfgwOp,
        ScfgwiOp,
        FrepOuterOp,
        FrepInnerOp,
        FrepYieldOp,
        ReadOp,
        WriteOp,
        GetStreamOp,
        DMSourceOp,
        DMDestinationOp,
        DMStrideOp,
        DMRepOp,
        DMCopyOp,
        DMCopyImmOp,
        DMStatOp,
        DMStatImmOp,
        VFMulSOp,
        VFAddSOp,
        VFCpkASSOp,
        VFMacSOp,
        VFSumSOp,
        VFAddHOp,
        VFMaxSOp,
    ],
    [],
)
