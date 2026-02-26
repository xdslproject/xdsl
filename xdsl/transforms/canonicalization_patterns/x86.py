from typing import cast

from xdsl.dialects import x86
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.x86.registers import X86RegisterType
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveRedundantDS_Mov(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: x86.DS_MovOp, rewriter: PatternRewriter) -> None:
        if op.destination.type == op.source.type and op.destination.type.is_allocated:
            rewriter.replace_op(op, (), (op.source,))


class RS_Add_Zero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: x86.RS_AddOp, rewriter: PatternRewriter) -> None:
        if (
            value := get_constant_value(op.source)
        ) is not None and value.value.data == 0:
            # The register would be updated in-place, so no need to move
            rewriter.replace_op(op, (), (op.register_in,))


class MS_Operation_ConstantOffset(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op, x86.ops.MS_Operation)
            and isinstance(add_op := op.memory.owner, x86.RS_AddOp)
            and ((value := get_constant_value(add_op.source)) is not None)
            # The addition is in-place, so we have to get the original value
            and isinstance(mov_op := add_op.register_in.owner, x86.DS_MovOp)
        ):
            op = cast(x86.ops.MS_Operation[X86RegisterType, X86RegisterType], op)
            new_offset = op.memory_offset.value.data + value.value.data
            rewriter.replace_op(op, type(op)(mov_op.source, op.source, new_offset))


class DM_Operation_ConstantOffset(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op, x86.ops.DM_Operation)
            and isinstance(add_op := op.memory.owner, x86.RS_AddOp)
            and ((value := get_constant_value(add_op.source)) is not None)
            # The addition is in-place, so we have to get the original value
            and isinstance(mov_op := add_op.register_in.owner, x86.DS_MovOp)
        ):
            op = cast(x86.ops.DM_Operation[X86RegisterType, X86RegisterType], op)
            new_offset = op.memory_offset.value.data + value.value.data
            rewriter.replace_op(
                op, type(op)(mov_op.source, new_offset, destination=op.destination.type)
            )


def get_constant_value(value: SSAValue) -> IntegerAttr | None:
    if not isinstance(value, OpResult):
        return

    if isinstance(value.op, x86.DS_MovOp):
        return get_constant_value(value.op.source)

    if isinstance(value.op, x86.DI_MovOp):
        return value.op.immediate
