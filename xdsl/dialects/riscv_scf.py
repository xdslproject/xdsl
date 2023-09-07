"""
RISC-V SCF dialect
"""
from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from xdsl.dialects.riscv import IntRegisterType, RISCVRegisterType
from xdsl.dialects.utils import (
    parse_assignment,
    parse_return_op_like,
    print_assignment,
    print_return_op_like,
)
from xdsl.ir import Attribute, Dialect
from xdsl.irdl import (
    Block,
    IRDLOperation,
    Operand,
    Operation,
    Region,
    SSAValue,
    VarOperand,
    VarOpResult,
    irdl_op_definition,
    operand_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator, SingleBlockImplicitTerminator
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "riscv_scf.yield"

    arguments: VarOperand = var_operand_def(RISCVRegisterType)

    # TODO circular dependency disallows this set of traits
    # tracked by gh issues https://github.com/xdslproject/xdsl/issues/1218
    # traits = frozenset([HasParent((For, If, ParallelOp, While)), IsTerminator()])
    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(operands=[[SSAValue.get(operand) for operand in operands]])

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, self.arguments)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs, args = parse_return_op_like(parser)
        op = cls(*args)
        op.attributes.update(attrs)
        return op


@irdl_op_definition
class ForOp(IRDLOperation):
    name = "riscv_scf.for"

    lb: Operand = operand_def(IntRegisterType)
    ub: Operand = operand_def(IntRegisterType)
    step: Operand = operand_def(IntRegisterType)

    iter_args: VarOperand = var_operand_def(RISCVRegisterType)

    res: VarOpResult = var_result_def(RISCVRegisterType)

    body: Region = region_def("single_block")

    traits = frozenset([SingleBlockImplicitTerminator(YieldOp)])

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
                f"Wrong number of block arguments, expected {len(self.iter_args)+1}, got "
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

    def print(self, printer: Printer):
        block = self.body.block
        index, *iter_args = block.args
        printer.print_string(" ")
        printer.print_ssa_value(index)
        printer.print(" : ")
        printer.print_attribute(index.type)
        printer.print_string(" = ")
        printer.print_ssa_value(self.lb)
        printer.print_string(" to ")
        printer.print_ssa_value(self.ub)
        printer.print_string(" step ")
        printer.print_ssa_value(self.step)
        printer.print_string(" ")
        if iter_args:
            printer.print_string("iter_args(")
            printer.print_list(
                zip(iter_args, self.iter_args),
                lambda pair: print_assignment(printer, *pair),
            )
            printer.print_string(") -> (")
            printer.print_list((a.type for a in iter_args), printer.print_attribute)
            printer.print_string(") ")
        printer.print_region(
            self.body, print_entry_block_args=False, print_empty_block=False
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        # Parse bounds
        index = parser.parse_argument(expect_type=False)
        parser.parse_characters(":")
        index_arg_type = parser.parse_type()
        parser.parse_characters("=")
        lb = parser.parse_operand()
        parser.parse_characters("to")
        ub = parser.parse_operand()
        parser.parse_characters("step")
        step = parser.parse_operand()

        # Parse iteration arguments
        pos = parser.pos
        iter_args: list[Parser.Argument] = []
        iter_arg_unresolved_operands: list[UnresolvedOperand] = []
        iter_arg_types: list[Attribute] = []
        if parser.parse_optional_characters("iter_args"):
            for iter_arg, iter_arg_operand in parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, lambda: parse_assignment(parser)
            ):
                iter_args.append(iter_arg)
                iter_arg_unresolved_operands.append(iter_arg_operand)
            parser.parse_characters("->")
            iter_arg_types = parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, parser.parse_attribute
            )

        iter_arg_operands = parser.resolve_operands(
            iter_arg_unresolved_operands, iter_arg_types, pos
        )

        # Set block argument types
        index.type = index_arg_type
        for iter_arg, iter_arg_type in zip(iter_args, iter_arg_types):
            iter_arg.type = iter_arg_type

        # Parse body
        body = parser.parse_region((index, *iter_args))
        if not body.block.ops:
            assert not iter_args, "Cannot create implicit yield with arguments"
            body.block.add_op(YieldOp())

        return cls(lb, ub, step, iter_arg_operands, body)


@irdl_op_definition
class WhileOp(IRDLOperation):
    name = "riscv_scf.while"
    arguments: VarOperand = var_operand_def(RISCVRegisterType)

    res: VarOpResult = var_result_def(RISCVRegisterType)
    before_region: Region = region_def()
    after_region: Region = region_def()

    def __init__(
        self,
        arguments: Sequence[SSAValue | Operation],
        result_types: Sequence[RISCVRegisterType],
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


@irdl_op_definition
class ConditionOp(IRDLOperation):
    name = "riscv_scf.condition"
    cond: Operand = operand_def(IntRegisterType)
    arguments: VarOperand = var_operand_def(RISCVRegisterType)

    traits = frozenset([HasParent(WhileOp), IsTerminator()])

    def __init__(self, cond: SSAValue | Operation, *output_ops: SSAValue | Operation):
        super().__init__(operands=[cond, output_ops])


RISCV_Scf = Dialect(
    [
        YieldOp,
        ForOp,
        WhileOp,
        ConditionOp,
    ],
    [],
)
