from __future__ import annotations

from dataclasses import dataclass

from xdsl.backend.liveness import LivenessContext
from xdsl.backend.x86.arch import X86Arch
from xdsl.builder import Builder
from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import InsertPoint


@dataclass
class LegalizationContext(LivenessContext):
    """
    Helper to ensure that registers can be allocated in the input, inserting a move if
    not.
    """

    arch: X86Arch
    builder: Builder

    def handle_live_inout(
        self, op: Operation, value: SSAValue, *, duplicate_inout: bool = False
    ) -> SSAValue:
        return self.arch.move_value_to_unallocated(
            value,
            self.builder,
            value_type=None,
            insertion_point=InsertPoint.before(op),
        )
