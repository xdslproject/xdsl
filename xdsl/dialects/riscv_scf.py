"""
RISC-V SCF dialect
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Generator, Sequence
from typing import cast

from typing_extensions import Self

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_allocator import BlockAllocator
from xdsl.backend.register_type import RegisterType
from xdsl.dialects.riscv import IntRegisterType, RISCVRegisterType
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_for_op_like,
    print_for_op_like,
)
from xdsl.ir import Attribute, Dialect
from xdsl.irdl import (
    Block,
    IRDLOperation,
    Operation,
    Region,
    SSAValue,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsTerminator,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class YieldOp(AbstractYieldOperation[RISCVRegisterType]):
    name = "riscv_scf.yield"

    traits = lazy_traits_def(
        lambda: (IsTerminator(), HasParent(WhileOp, ForRofOperation))
    )


class ForRofOperation(RegisterAllocatableOperation, IRDLOperation, ABC):
    lb = operand_def(IntRegisterType)
    ub = operand_def(IntRegisterType)
    step = operand_def(IntRegisterType)

    iter_args = var_operand_def(RISCVRegisterType)

    res = var_result_def(RISCVRegisterType)

    body = region_def("single_block")

    traits = traits_def(SingleBlockImplicitTerminator(YieldOp))

    def __init__(
        self,
        lb: SSAValue | Operation,
        ub: SSAValue | Operation,
        step: SSAValue | Operation,
        iter_args: Sequence[SSAValue | Operation],
        body: Region | Sequence[Operation] | Sequence[Block] | Block,
    ):
        if isinstance(body, Block):
            body = [body]

        super().__init__(
            operands=[lb, ub, step, iter_args],
            result_types=[[SSAValue.get(a).type for a in iter_args]],
            regions=[body],
        )

    def verify_(self):
        if (len(self.iter_args) + 1) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args) + 1}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments."
            )
        if self.body.block.args and (iter_var := self.body.block.args[0]):
            if not isinstance(iter_var.type, IntRegisterType):
                raise VerifyException(
                    f"The first block argument of the body is of type {iter_var.type}"
                    " instead of riscv.IntRegisterType"
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
                    f"The riscv_scf.for must yield its carried variables."
                )
            for iter_arg, yield_arg in zip(self.iter_args, yieldop.arguments):
                if iter_arg.type != yield_arg.type:
                    raise VerifyException(
                        f"Expected {iter_arg.type}, got {yield_arg.type}. The "
                        f"riscv_scf.for's riscv_scf.yield must match carried"
                        f"variables types."
                    )

    def iter_used_registers(self) -> Generator[RegisterType, None, None]:
        # We know that all the registers for the inputs and outputs are the same, and
        # that these registers will have been iterated earlier in the IR.
        yield from ()

    def allocate_registers(self, allocator: BlockAllocator) -> None:
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = allocator.live_ins_per_block[self.body.block]
        for live_in in live_ins:
            allocator.allocate_value(live_in)

        yield_op = self.body.block.last_op
        assert yield_op is not None, (
            "last op of riscv_scf.ForOp is guaranteed to be riscv_scf.Yield"
        )
        block_args = self.body.block.args

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args[1:], self.iter_args, yield_op.operands, self.results
        ):
            allocator.allocate_values_same_reg(
                (block_arg, operand, yield_operand, op_result)
            )

        # Induction variable
        allocator.allocate_value(block_args[0])

        # Step and ub are used throughout loop
        allocator.allocate_value(self.ub)
        allocator.allocate_value(self.step)

        # Reserve the loop carried variables for allocation within the body
        regs = self.iter_args.types
        assert all(isinstance(reg, RISCVRegisterType) for reg in regs)
        regs = cast(tuple[RISCVRegisterType, ...], regs)
        with allocator.available_registers.reserve_registers(regs):
            allocator.allocate_block(self.body.block)

        # lb is only used as an input to the loop, so free induction variable before
        # allocating lb to it in case it's not yet allocated
        allocator.free_value(self.body.block.args[0])
        allocator.allocate_value(self.lb)


@irdl_op_definition
class ForOp(ForRofOperation):
    """
    A for loop, counting up from lb to ub by step each iteration.
    """

    name = "riscv_scf.for"

    def print(self, printer: Printer):
        print_for_op_like(
            printer,
            self.lb,
            self.ub,
            self.step,
            self.iter_args,
            self.body,
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        lb, ub, step, iter_arg_operands, body = parse_for_op_like(parser)
        _, *iter_args = body.block.args

        for_op = cls(lb, ub, step, iter_arg_operands, body)

        if not iter_args:
            for trait in for_op.get_traits_of_type(SingleBlockImplicitTerminator):
                ensure_terminator(for_op, trait)

        return for_op


@irdl_op_definition
class RofOp(ForRofOperation):
    """
    Reverse Order For loop.

    MLIR's for loops have the constraint of always executing from lb to ub,
    so in order to express loops that count down from ub to lb, the rof op
    is needed.

    Rof has the semantics of going from ub to lb, decrementing by step each time.
    The implicit constraints are that lb < ub, and step > 0.

    In order to convert a for to a rof, one needs to switch lb and ub.
    (for the normalized case that (ub - lb) % step == 0)
    """

    name = "riscv_scf.rof"

    def print(self, printer: Printer):
        print_for_op_like(
            printer,
            self.ub,
            self.lb,
            self.step,
            self.iter_args,
            self.body,
            bound_words=["down", "to"],
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        ub, lb, step, iter_arg_operands, body = parse_for_op_like(
            parser, bound_words=["down", "to"]
        )
        _, *iter_args = body.block.args

        rof_op = cls(lb, ub, step, iter_arg_operands, body)

        if not iter_args:
            for trait in rof_op.get_traits_of_type(SingleBlockImplicitTerminator):
                ensure_terminator(rof_op, trait)

        return rof_op


@irdl_op_definition
class WhileOp(IRDLOperation):
    name = "riscv_scf.while"
    arguments = var_operand_def(RISCVRegisterType)

    res = var_result_def(RISCVRegisterType)
    before_region = region_def()
    after_region = region_def()

    def __init__(
        self,
        arguments: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        before_region: Region | Sequence[Operation] | Sequence[Block],
        after_region: Region | Sequence[Operation] | Sequence[Block],
    ):
        super().__init__(
            operands=[arguments],
            result_types=[result_types],
            regions=[before_region, after_region],
        )

    # TODO verify dependencies between riscv_scf.condition, riscv_scf.yield and the regions
    def verify_(self):
        for idx, (block_arg, arg) in enumerate(
            zip(
                self.before_region.block.args,
                self.arguments,
                strict=True,
            )
        ):
            if block_arg.type != arg.type:
                raise VerifyException(
                    f"Block arguments at {idx} has wrong type,"
                    f" expected {arg.type},"
                    f" got {block_arg.type}"
                )

        for idx, (block_arg, res) in enumerate(
            zip(
                self.after_region.block.args,
                self.res,
                strict=True,
            )
        ):
            if block_arg.type != res.type:
                raise VerifyException(
                    f"Block arguments at {idx} has wrong type,"
                    f" expected {res.type},"
                    f" got {block_arg.type}"
                )

    @staticmethod
    def _print_pair(printer: Printer, pair: tuple[SSAValue, SSAValue]):
        printer.print_ssa_value(pair[0])
        printer.print_string(" = ")
        printer.print_ssa_value(pair[1])

    def print(self, printer: Printer):
        printer.print_string(" (")
        block_args = self.before_region.block.args
        printer.print_list(
            zip(block_args, self.arguments, strict=True),
            lambda pair: self._print_pair(printer, pair),
        )
        printer.print_string(") : ")
        printer.print_operation_type(self)
        printer.print_string(" ")
        printer.print_region(self.before_region, print_entry_block_args=False)
        printer.print_string(" do ")
        printer.print_region(self.after_region)
        if self.attributes:
            printer.print_op_attributes(self.attributes, print_keyword=True)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        def parse_assignment():
            arg = parser.parse_argument(expect_type=False)
            parser.parse_punctuation("=")
            operand = parser.parse_unresolved_operand()
            return arg, operand

        tuples = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN,
            parse_assignment,
        )

        parser.parse_punctuation(":")
        type_pos = parser.pos
        function_type = parser.parse_function_type()

        if len(tuples) != len(function_type.inputs.data):
            parser.raise_error(
                f"Mismatch between block argument count ({len(tuples)}) and operand count ({len(function_type.inputs.data)})",
                type_pos,
                parser.pos,
            )

        block_args = tuple(
            block_arg.resolve(t)
            for ((block_arg, _), t) in zip(
                tuples, function_type.inputs.data, strict=True
            )
        )

        arguments = tuple(
            parser.resolve_operand(operand, t)
            for ((_, operand), t) in zip(tuples, function_type.inputs.data, strict=True)
        )

        before_region = parser.parse_region(block_args)
        parser.parse_characters("do")
        after_region = parser.parse_region()

        attrs = parser.parse_optional_attr_dict_with_keyword()

        op = cls(arguments, function_type.outputs.data, before_region, after_region)

        if attrs is not None:
            op.attributes |= attrs.data

        return op


@irdl_op_definition
class ConditionOp(IRDLOperation):
    name = "riscv_scf.condition"
    cond = operand_def(IntRegisterType)
    arguments = var_operand_def(RISCVRegisterType)

    traits = traits_def(HasParent(WhileOp), IsTerminator())

    def __init__(self, cond: SSAValue | Operation, *output_ops: SSAValue | Operation):
        super().__init__(operands=[cond, output_ops])

    def print(self, printer: Printer):
        printer.print_string("(")
        printer.print_ssa_value(self.cond)
        printer.print_string(" : ")
        printer.print_attribute(self.cond.type)
        printer.print_string(") ")
        if self.attributes:
            printer.print_op_attributes(self.attributes)
        if self.arguments:
            printer.print_string(" ")
            printer.print_list(self.arguments, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(
                self.arguments, lambda val: printer.print_attribute(val.type)
            )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_punctuation("(")
        unresolved_cond = parser.parse_unresolved_operand("cond expected")
        parser.parse_punctuation(":")
        cond_type = parser.parse_type()
        parser.parse_punctuation(")")
        cond = parser.resolve_operand(unresolved_cond, cond_type)
        attrs = parser.parse_optional_attr_dict()

        # scf.condition is a terminator, so the list of arguments cannot be confused with
        # the results of a hypothetical operation on the next line.
        pos = parser.pos
        unresolved_arguments = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )
        if unresolved_arguments is not None:
            parser.parse_punctuation(":")
            types = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_type
            )
            arguments = parser.resolve_operands(unresolved_arguments, types, pos)
        else:
            arguments: Sequence[SSAValue] = ()

        op = cls(cond, *arguments)
        op.attributes = attrs
        return op


RISCV_Scf = Dialect(
    "riscv_scf",
    [
        YieldOp,
        ForOp,
        RofOp,
        WhileOp,
        ConditionOp,
    ],
    [],
)
