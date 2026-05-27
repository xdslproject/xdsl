from dataclasses import dataclass

from xdsl.backend.register_type import RegisterResource, RegisterType
from xdsl.context import Context
from xdsl.dialects import asm, builtin, llvm
from xdsl.dialects.x86 import ops as x86_ops
from xdsl.dialects.x86.registers import X86RegisterType
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import MemoryEffectKind, get_effects
from xdsl.utils.exceptions import DiagnosticException


@dataclass(frozen=True)
class ExtractedAsmRegionData:
    """
    Extracted inline-assembly payload from an `asm.region`.

    The fields map directly to the pieces needed to build `llvm.inline_asm`:
    emitted assembly lines, ordered output/input constraints, and derived clobbers.
    """

    asm_lines: tuple[str, ...]
    """Assembly lines emitted by x86 ops in region order, excluding non-emitting ops."""

    output_registers: tuple[X86RegisterType, ...]
    """Output registers in yielded-value order."""

    input_registers: tuple[X86RegisterType, ...]
    """Input registers in entry-argument order."""

    used_registers: tuple[X86RegisterType, ...]
    """Allocated registers touched by body operations, in first-seen order."""

    read_registers: frozenset[X86RegisterType]
    """Allocated registers read by body operations according to memory effects."""

    written_registers: frozenset[X86RegisterType]
    """Allocated registers written by body operations according to memory effects."""

    @staticmethod
    def _constraint(register: X86RegisterType, prefix: str) -> str:
        return f"{prefix}{{{_require_allocated_register_name(register, 'inline asm constraints')}}}"

    @property
    def constraints(self) -> tuple[str, ...]:
        """Constraints in LLVM inline-asm order: outputs, inputs, clobbers."""
        remaining_input_registers = list(self.input_registers)
        output_constraints: list[str] = []
        for register in self.output_registers:
            if (
                register in remaining_input_registers
                and register in self.read_registers
                and register in self.written_registers
            ):
                output_constraints.append(self._constraint(register, "+"))
                remaining_input_registers.remove(register)
            else:
                output_constraints.append(self._constraint(register, "="))

        input_constraints = tuple(
            self._constraint(register, "") for register in remaining_input_registers
        )
        boundary_registers = set(self.output_registers) | set(remaining_input_registers)
        clobber_constraints = tuple(
            self._constraint(register, "~")
            for register in self.used_registers
            if register not in boundary_registers
        )
        return tuple(output_constraints) + input_constraints + clobber_constraints


def _require_allocated_register_name(register: RegisterType, context: str) -> str:
    register_name = register.register_name.data
    if not register_name:
        raise DiagnosticException(
            f"asm-to-llvm requires allocated x86 registers in {context}"
        )
    return register_name


def _value_register(value: SSAValue, context: str) -> X86RegisterType:
    register_type = value.type
    if not isinstance(register_type, X86RegisterType):
        raise DiagnosticException(
            f"asm-to-llvm currently supports only x86 register-typed values in {context}"
        )
    _require_allocated_register_name(register_type, context)
    return register_type


def _extract_asm_region_data(
    op: asm.RegionOp, terminator: asm.YieldOp
) -> ExtractedAsmRegionData:
    block = op.body.block

    asm_lines: list[str] = []
    used_registers: list[X86RegisterType] = []
    seen_used_registers: set[X86RegisterType] = set()
    read_registers: set[X86RegisterType] = set()
    written_registers: set[X86RegisterType] = set()

    for body_op in block.ops:
        if isinstance(body_op, asm.YieldOp):
            continue
        if not isinstance(body_op, x86_ops.X86AsmOperation):
            raise DiagnosticException(
                f"asm-to-llvm supports only x86 ops in asm.region body, got {body_op.name}"
            )

        line = body_op.assembly_line()
        if line is not None:
            asm_lines.append(line.strip())

        effects = get_effects(body_op) or set()
        for effect in effects:
            if not isinstance(resource := effect.resource, RegisterResource):
                continue
            register = resource.register
            if not isinstance(register, X86RegisterType) or not register.is_allocated:
                continue
            if register not in seen_used_registers:
                seen_used_registers.add(register)
                used_registers.append(register)
            if effect.kind is MemoryEffectKind.READ:
                read_registers.add(register)
            elif effect.kind is MemoryEffectKind.WRITE:
                written_registers.add(register)

    output_registers = tuple(
        _value_register(yielded, "asm.yield operands")
        for yielded in terminator.arguments
    )
    input_registers = tuple(
        _value_register(arg, "asm.region entry block arguments") for arg in block.args
    )

    return ExtractedAsmRegionData(
        asm_lines=tuple(asm_lines),
        output_registers=output_registers,
        input_registers=input_registers,
        used_registers=tuple(used_registers),
        read_registers=frozenset(read_registers),
        written_registers=frozenset(written_registers),
    )


@dataclass(frozen=True)
class AsmRegionToLLVMPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: asm.RegionOp, rewriter: PatternRewriter) -> None:
        if len(op.body.blocks) != 1:
            raise DiagnosticException(
                "asm-to-llvm currently supports only single-block asm.region"
            )

        block = op.body.block
        terminator = block.last_op
        assert isinstance(terminator, asm.YieldOp)
        if len(terminator.arguments) > 1:
            raise DiagnosticException(
                "asm-to-llvm currently supports only up to one yielded value"
            )

        for arg in block.args:
            if not isinstance(arg.type, X86RegisterType):
                raise DiagnosticException(
                    "asm-to-llvm currently supports only x86 register-typed entry args"
                )

        for body_op in block.ops:
            if isinstance(body_op, asm.YieldOp):
                for yielded in body_op.arguments:
                    if not isinstance(yielded.type, X86RegisterType):
                        raise DiagnosticException(
                            "asm-to-llvm currently supports only x86 register-typed yields"
                        )
                continue
            if not isinstance(body_op, x86_ops.X86AsmOperation):
                raise DiagnosticException(
                    f"asm-to-llvm supports only x86 ops in asm.region body, got {body_op.name}"
                )

        extracted = _extract_asm_region_data(op, terminator)
        asm_string = "\n".join(extracted.asm_lines)
        constraints = ",".join(extracted.constraints)
        result_types = [op.results[0].type] if len(op.results) == 1 else []
        inline_asm_op = llvm.InlineAsmOp(
            asm_string=asm_string,
            constraints=constraints,
            operands=op.operands,
            res_types=result_types,
            asm_dialect=llvm.ASM_DIALECT_KEY_BY_NAME["intel"],
            has_side_effects=True,
        )
        rewriter.replace_op(
            op,
            inline_asm_op,
            new_results=inline_asm_op.results,
        )


class AsmToLLVMPass(ModulePass):
    """
    Convert x86-only asm.region operations to llvm.inline_asm.
    """

    name = "asm-to-llvm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AsmRegionToLLVMPattern()).rewrite_module(op)
