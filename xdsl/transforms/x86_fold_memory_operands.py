"""
Fold vector loads into the memory operand of FMA instructions.

x86 FMA instructions can take one of their multiply operands directly from
memory instead of from a register:

    vmovupd     ymm2, [rdi]
    vfmadd231pd ymm0, ymm1, ymm2      ->    vfmadd231pd ymm0, ymm1, [rdi]

This saves an instruction and, more importantly for a register-starved kernel
like matmul, frees the vector register that was only holding the loaded value.

AVX-512 additionally supports an embedded broadcast on the memory operand, so a
broadcast-load feeding an FMA collapses the same way:

    vbroadcastsd zmm2, [rdi]
    vfmadd231pd  zmm0, zmm1, zmm2     ->    vfmadd231pd zmm0, zmm1, [rdi]{1to8}

The embedded broadcast needs EVEX encoding, so that fold is gated on an AVX-512
target. The plain memory-operand fold is available under VEX and so applies to
SSE and AVX2 as well.
"""

from dataclasses import dataclass
from typing import TypeAlias, cast

from xdsl.backend.x86.arch import AVX512Arch, X86Arch
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86.ops import (
    DM_VbroadcastsdOp,
    DM_VbroadcastssOp,
    DM_VmovapdOp,
    DM_VmovapsOp,
    DM_VmovupdOp,
    DM_VmovupsOp,
    RSM_Vfmadd231pdOp,
    RSM_Vfmadd231psOp,
    RSS_Vfmadd231pdOp,
    RSS_Vfmadd231psOp,
)
from xdsl.dialects.x86.registers import X86VectorRegisterType
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import MemoryEffectKind, get_effects

FoldableLoad: TypeAlias = (
    DM_VmovupdOp
    | DM_VmovapdOp
    | DM_VmovupsOp
    | DM_VmovapsOp
    | DM_VbroadcastsdOp
    | DM_VbroadcastssOp
)
"""Loads whose result can become the memory operand of an FMA."""


def _writes_memory(op: Operation) -> bool:
    """
    Conservatively report whether `op` may write to memory.

    An operation with unknown effects (`get_effects` returning None) is treated
    as writing, since we cannot prove that it does not.
    """
    effects = get_effects(op)
    if effects is None:
        return True
    return any(e.kind == MemoryEffectKind.WRITE for e in effects)


def _safe_to_sink(load: Operation, user: Operation) -> bool:
    """
    Check that `load` can be sunk to the position of `user`.

    Folding the load into `user` delays the memory read until `user` executes,
    so it is only valid if nothing in between may write to memory. Both
    operations must live in the same block, which is the case for the
    straight-line code the x86 backend produces after lowering.
    """
    block = load.parent_block()
    if block is None or user.parent_block() is not block:
        return False

    current = load.next_op
    while current is not None and current is not user:
        if _writes_memory(current):
            return False
        current = current.next_op

    # `user` must actually come after `load` in the block.
    return current is user


def _load_to_fold(
    value: SSAValue,
    allowed: tuple[type[FoldableLoad], ...],
    user: Operation,
) -> FoldableLoad | None:
    """
    Return the load defining `value` if it can be folded into `user`, else None.
    """
    load = value.owner
    if not isinstance(load, allowed):
        return None
    # Folding would duplicate the memory access if the loaded value is read
    # elsewhere, which is a pessimisation rather than an optimisation.
    if not value.has_one_use():
        return None
    if not _safe_to_sink(load, user):
        return None
    return load


@dataclass
class FoldLoadIntoFMA(RewritePattern):
    """
    Fold a vector load or broadcast-load into the memory operand of an FMA.
    """

    arch: X86Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: RSS_Vfmadd231pdOp | RSS_Vfmadd231psOp,
        rewriter: PatternRewriter,
    ) -> None:
        rsm_type: type[RSM_Vfmadd231pdOp | RSM_Vfmadd231psOp]
        direct_loads: tuple[type[FoldableLoad], ...]
        broadcast_loads: tuple[type[FoldableLoad], ...]

        if isinstance(op, RSS_Vfmadd231pdOp):
            rsm_type = RSM_Vfmadd231pdOp
            direct_loads = (DM_VmovupdOp, DM_VmovapdOp)
            broadcast_loads = (DM_VbroadcastsdOp,)
        else:
            rsm_type = RSM_Vfmadd231psOp
            direct_loads = (DM_VmovupsOp, DM_VmovapsOp)
            broadcast_loads = (DM_VbroadcastssOp,)

        # The embedded broadcast modifier needs EVEX encoding, which is an
        # AVX-512 feature. AVX512VL extends EVEX to xmm and ymm operands, so
        # this is a property of the target rather than of the vector width.
        candidates: list[tuple[tuple[type[FoldableLoad], ...], bool]] = [
            (direct_loads, False)
        ]
        if isinstance(self.arch, AVX512Arch):
            candidates.append((broadcast_loads, True))

        # `register_in += source1 * source2`. Multiplication is commutative, so
        # either source may become the memory operand and the other stays in a
        # register. Try source2 first, so the common case rewrites without
        # reordering the operands.
        for allowed, broadcast in candidates:
            for mem_operand, reg_operand in (
                (op.source2, op.source1),
                (op.source1, op.source2),
            ):
                load = _load_to_fold(mem_operand, allowed, op)
                if load is None:
                    continue

                folded = rsm_type(
                    cast(SSAValue[X86VectorRegisterType], op.register_in),
                    reg_operand,
                    load.memory,
                    load.memory_offset.value.data,
                    broadcast=broadcast,
                    comment=op.comment,
                    register_out=op.register_out.type,
                )
                # The fused instruction takes the FMA's position, which sinks
                # the memory read to that point; `_safe_to_sink` has already
                # checked that nothing in between writes memory. The load's
                # address operand dominated the load and so dominates here too.
                rewriter.replace(op, [folded])
                # The load's only use was the FMA we just replaced.
                rewriter.erase(load)
                return


@dataclass(frozen=True)
class X86FoldMemoryOperands(ModulePass):
    """
    Folds vector loads into the memory operands of FMA instructions.

    Run before register allocation, so that the vector registers freed by the
    fold are available to the allocator.
    """

    name = "x86-fold-memory-operands"

    arch: str

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        arch = X86Arch.arch_for_name(self.arch)
        PatternRewriteWalker(
            GreedyRewritePatternApplier([FoldLoadIntoFMA(arch)]),
            apply_recursively=False,
        ).rewrite_module(op)
