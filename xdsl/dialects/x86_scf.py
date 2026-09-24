from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import cast

from typing_extensions import Self

from xdsl.backend.liveness import LivenessContext
from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.backend.register_allocator import BlockAllocator, live_ins_per_block
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_for_op_like,
    print_for_op_like,
)
from xdsl.dialects.x86.ops import SI32, X86HasRegisterConstraints
from xdsl.dialects.x86.registers import GeneralRegisterType, X86RegisterType
from xdsl.ir import Dialect
from xdsl.irdl import (
    AttrSizedOperandSegments,
    Block,
    Operation,
    Region,
    SSAValue,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class YieldOp(AbstractYieldOperation[X86RegisterType]):
    name = "x86_scf.yield"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(ForRofOperation),
            NoMemoryEffect(),
        )
    )


class ForRofOperation(X86HasRegisterConstraints, ABC):
    """
    Loops where `start` initializes the IV and `stop` is the exclusive termination
    bound.
    Both `stop` and `step` may be immediates, whereas `start` must be in a register,
    which will be updated throughout loop iteration.
    """

    start = operand_def(GeneralRegisterType)
    stop_val = opt_operand_def(GeneralRegisterType)
    stop_attr = opt_prop_def(IntegerAttr[SI32])
    step_val = opt_operand_def(GeneralRegisterType)
    step_attr = opt_prop_def(IntegerAttr[SI32])

    iter_args = var_operand_def(X86RegisterType)

    iv_end = result_def(GeneralRegisterType)
    """
    Induction variable on exit, in the same register as `start`.
    For a zero-trip loop this is `start`, otherwise it is the value after the final
    increment (`for`) or decrement (`rof`), which may overshoot `stop`.
    """

    res = var_result_def(X86RegisterType)

    body = region_def("single_block")

    traits = traits_def(SingleBlockImplicitTerminator(YieldOp), RecursiveMemoryEffect())
    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    @property
    def stop(self) -> IntegerAttr[SI32] | SSAValue:
        """Static termination bound (typed integer) or dynamic register SSA value."""
        match (stop_attr := self.stop_attr, stop_val := self.stop_val):
            case (None, None):
                raise ValueError("Exactly one of stop_attr or stop_val must be set")
            case (None, _):
                return stop_val
            case (_, None):
                return stop_attr
            case (_, _):
                raise ValueError("Exactly one of stop_attr or stop_val must be set")

    @property
    def step(self) -> IntegerAttr[SI32] | SSAValue:
        """Static step (typed integer) or dynamic register SSA value."""
        match (step_attr := self.step_attr, step_val := self.step_val):
            case (None, None):
                raise ValueError("Exactly one of step_attr or step_val must be set")
            case (None, _):
                return step_val
            case (_, None):
                return step_attr
            case (_, _):
                raise ValueError("Exactly one of step_attr or step_val must be set")

    def __init__(
        self,
        start: SSAValue | Operation,
        stop: SSAValue | Operation | IntegerAttr,
        step: SSAValue | Operation | IntegerAttr,
        iter_args: Sequence[SSAValue],
        body: Region | Sequence[Operation] | Sequence[Block] | Block | None = None,
    ):
        start = SSAValue.get(start)
        if body is None:
            body = Region(
                Block(
                    arg_types=(start.type, *(iter_arg.type for iter_arg in iter_args))
                )
            )

        if isinstance(body, Block):
            body = [body]

        if isinstance(stop, IntegerAttr):
            stop_attr, stop_val = stop, None
        else:
            stop_attr, stop_val = None, stop

        if isinstance(step, IntegerAttr):
            step_attr = step
            step_val = None
        else:
            step_attr = None
            step_val = step

        super().__init__(
            operands=[start, stop_val, step_val, iter_args],
            properties={"stop_attr": stop_attr, "step_attr": step_attr},
            result_types=[start.type, [SSAValue.get(a).type for a in iter_args]],
            regions=[body],
        )

    def verify_(self):
        try:
            _ = self.stop
            _ = self.step
        except ValueError as exc:
            raise VerifyException(exc.args[0])

        if (len(self.iter_args) + 1) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args) + 1}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments."
            )
        if self.body.block.args and (iter_var := self.body.block.args[0]):
            if not isinstance(iter_var.type, GeneralRegisterType):
                raise VerifyException(
                    f"The first block argument of the body is of type {iter_var.type}"
                    " instead of x86 GeneralRegisterType"
                )
            start = self.start
            if iter_var.type != start.type:
                raise VerifyException(
                    f"Expected induction var to be same type as start, "
                    f"got {iter_var.type} and {start.type}"
                )
            if iter_var.type != self.iv_end.type:
                raise VerifyException(
                    f"Expected induction var to be same type as iv_end result, "
                    f"got {iter_var.type} and {self.iv_end.type}"
                )
        for idx, (arg, block_arg) in enumerate(
            zip(self.iter_args, self.body.block.args[1:])
        ):
            if block_arg.type != arg.type:
                raise VerifyException(
                    f"Block argument {idx + 1} has wrong type, expected {arg.type}, "
                    f"got {block_arg.type}. Arguments after the "
                    f"induction variable must match the carried variables."
                )
        if len(self.body.ops) > 0 and isinstance(
            yieldop := self.body.block.last_op, YieldOp
        ):
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The {self.name} must yield its carried variables."
                )
            for iter_arg, yield_arg in zip(self.iter_args, yieldop.arguments):
                if iter_arg.type != yield_arg.type:
                    raise VerifyException(
                        f"Expected {iter_arg.type}, got {yield_arg.type}. The "
                        f"{self.name}'s {YieldOp.name} must match carried "
                        f"variables types."
                    )

    def allocate_registers(self, allocator: BlockAllocator) -> None:
        """Allocate loop-carried and IV registers, then the body under those reservations."""
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = allocator.live_ins_per_block[self.body.block]
        for live_in in live_ins:
            allocator.allocate_value(live_in)

        yield_op = self.body.block.last_op
        assert yield_op is not None, (
            f"last op of {self.name} is guaranteed to be {YieldOp.name}"
        )
        block_args = self.body.block.args

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args[1:], self.iter_args, yield_op.operands, self.res, strict=True
        ):
            allocator.allocate_values_same_reg(
                (block_arg, operand, yield_operand, op_result)
            )

        allocator.allocate_values_same_reg((block_args[0], self.start, self.iv_end))

        # stop and step are used throughout loop when dynamic
        if self.stop_val is not None:
            allocator.allocate_value(self.stop_val)
        if self.step_val is not None:
            allocator.allocate_value(self.step_val)

        # Reserve the loop carried variables for allocation within the body
        regs = self.iter_args.types
        assert all(isinstance(reg, X86RegisterType) for reg in regs)
        regs = cast(tuple[X86RegisterType, ...], regs)
        with allocator.available_registers.reserve_registers(regs):
            allocator.allocate_block(self.body.block)

    def get_register_constraints(self) -> RegisterConstraints:
        """`start` and each iter_arg are inout; dynamic `stop`/`step` are in-only."""
        ins = tuple(
            value for value in (self.stop_val, self.step_val) if value is not None
        )
        inouts = tuple(zip(self.iter_args, self.res, strict=True))
        return RegisterConstraints(ins, (), ((self.start, self.iv_end), *inouts))

    def _body_live_outs(self, live_after: AbstractSet[SSAValue]) -> set[SSAValue]:
        """
        Values live at the end of the loop body.
        """
        block = self.body.block
        res = set(live_after)
        # The body runs repeatedly, so every value defined outside it and used inside
        # must survive a whole iteration.
        res.update(live_ins_per_block(block)[block])
        if self.stop_val is not None:
            res.add(self.stop_val)
        if self.step_val is not None:
            res.add(self.step_val)
        # The induction variable is a block argument rather than a live-in, but the
        # loop reads it on the back edge to compute the next value.
        res.add(block.args[0])
        return res

    def update_liveness(self, ctx: LivenessContext) -> None:
        # Create a new context to use inside the loop
        body_ctx = ctx.copy(self._body_live_outs(ctx.alive))
        body_ctx.process_block(self.body.block)
        # Update the outer context with all the values that are alive coming into the
        # body
        ctx.alive.update(body_ctx.alive)
        # HasRegisterConstraints default implementation
        super().update_liveness(ctx)


@irdl_op_definition
class ForOp(ForRofOperation):
    """
    A for loop over [start, stop), incrementing by a positive step.
    If start >= stop the body does not execute.
    """

    name = "x86_scf.for"

    def print(self, printer: Printer):
        print_for_op_like(
            printer,
            self.start,
            self.stop,
            self.step,
            self.iter_args,
            self.body,
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        start, stop, step, iter_arg_operands, body = parse_for_op_like(
            parser, allow_static_stop=True, allow_static_step=True
        )
        _, *iter_args = body.block.args

        for_op = cls(start, stop, step, iter_arg_operands, body)

        if not iter_args:
            for trait in for_op.get_traits_of_type(SingleBlockImplicitTerminator):
                ensure_terminator(for_op, trait)

        return for_op


@irdl_op_definition
class RofOp(ForRofOperation):
    """
    A reverse loop over (stop, start], decrementing by a positive step.
    If start <= stop the body does not execute.
    """

    name = "x86_scf.rof"

    def print(self, printer: Printer):
        print_for_op_like(
            printer,
            self.start,
            self.stop,
            self.step,
            self.iter_args,
            self.body,
            bound_words=["down", "to"],
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        start, stop, step, iter_arg_operands, body = parse_for_op_like(
            parser,
            bound_words=["down", "to"],
            allow_static_stop=True,
            allow_static_step=True,
        )
        _, *iter_args = body.block.args

        rof_op = cls(start, stop, step, iter_arg_operands, body)

        if not iter_args:
            for trait in rof_op.get_traits_of_type(SingleBlockImplicitTerminator):
                ensure_terminator(rof_op, trait)

        return rof_op


X86_Scf = Dialect(
    "x86_scf",
    [
        ForOp,
        RofOp,
        YieldOp,
    ],
)
