from __future__ import annotations

from typing import Annotated, List

from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import AnyAttr, Operand, VarOperand, VarOpResult, irdl_op_definition
from xdsl.utils.exceptions import VerifyException

# TODO custom syntax


@irdl_op_definition
class If(Operation):
    name: str = "scf.if"
    output: Annotated[VarOpResult, AnyAttr()]
    cond: Annotated[Operand, IntegerType.from_width(1)]

    true_region: Region
    # TODO this should be optional under certain conditions
    false_region: Region

    @staticmethod
    def get(
        cond: SSAValue | Operation,
        return_types: List[Attribute],
        true_region: Region | List[Block] | List[Operation],
        false_region: Region | List[Block] | List[Operation],
    ) -> If:
        return If.build(
            operands=[cond],
            result_types=[return_types],
            regions=[true_region, false_region],
        )


@irdl_op_definition
class Yield(Operation):
    name: str = "scf.yield"
    arguments: Annotated[VarOperand, AnyAttr()]

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(
            operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class Condition(Operation):
    name: str = "scf.condition"
    cond: Annotated[Operand, IntegerType.from_width(1)]
    arguments: Annotated[VarOperand, AnyAttr()]

    @staticmethod
    def get(cond: SSAValue | Operation,
            *output_ops: SSAValue | Operation) -> Condition:
        return Condition.build(
            operands=[cond, [output for output in output_ops]])


@irdl_op_definition
class For(Operation):
    name: str = "scf.for"

    lb: Annotated[Operand, IndexType]
    ub: Annotated[Operand, IndexType]
    step: Annotated[Operand, IndexType]

    iter_args: Annotated[VarOperand, AnyAttr()]

    res: Annotated[VarOpResult, AnyAttr()]

    body: Region

    def verify_(self):
        if (len(self.iter_args) + 1) != len(self.body.blocks[0].args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args)+1}, got "
                f"{len(self.body.blocks[0].args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments.")
        if len(self.operands) < 3:
            raise VerifyException(
                "Expected at least 3 arguments. A scf.for must have a lower- and "
                "upper-bound, and a step.")
        for idx, arg in enumerate(self.iter_args):
            if self.body.blocks[0].args[idx + 1].typ != arg.typ:
                raise VerifyException(
                    f"Block arguments with wrong type, expected {arg.typ}, "
                    f"got {self.body.blocks[0].args[idx].typ}. Arguments after the "
                    f"induction variable must match the carried variables.")
        if len(self.iter_args) > 0:
            if len(self.body.blocks[0].ops) == 0 or not isinstance(
                    self.body.blocks[0].ops[-1], Yield):
                raise VerifyException(
                    "The scf.for's body does not end with a scf.yield. A scf.for loop "
                    "with loop-carried variables must yield their values at the end of "
                    "its body.")
        if len(self.body.blocks[0].ops) > 0 and isinstance(
                self.body.blocks[0].ops[-1], Yield):
            yieldop = self.body.blocks[0].ops[-1]
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The scf.for must yield its carried variables.")
            for idx, arg in enumerate(yieldop.arguments):
                if self.iter_args[idx].typ != arg.typ:
                    raise VerifyException(
                        f"Expected {self.iter_args[idx].typ}, got {arg.typ}. The "
                        f"scf.for's scf.yield must match carried variables types."
                    )

    @staticmethod
    def get(
        lb: SSAValue | Operation,
        ub: SSAValue | Operation,
        step: SSAValue | Operation,
        iter_args: List[SSAValue | Operation],
        body: Region | List[Operation] | List[Block] | Block,
    ) -> For:
        if isinstance(body, Block):
            body = [body]
        op = For.build(
            operands=[lb, ub, step, iter_args],
            result_types=[[SSAValue.get(a).typ for a in iter_args]],
            regions=[body],
        )
        return op


@irdl_op_definition
class While(Operation):
    name: str = "scf.while"
    arguments: Annotated[VarOperand, AnyAttr()]

    res: Annotated[VarOpResult, AnyAttr()]
    before_region: Region
    after_region: Region

    # TODO verify dependencies between scf.condition, scf.yield and the regions
    def verify_(self):
        for idx, arg in enumerate(self.arguments):
            if self.before_region.blocks[0].args[idx].typ != arg.typ:
                raise Exception(
                    f"Block arguments with wrong type, expected {arg.typ}, "
                    f"got {self.before_region.blocks[0].args[idx].typ}")

        for idx, res in enumerate(self.res):
            if self.after_region.blocks[0].args[idx].typ != res.typ:
                raise Exception(
                    f"Block arguments with wrong type, expected {res.typ}, "
                    f"got {self.after_region.blocks[0].args[idx].typ}")

    @staticmethod
    def get(
        operands: List[SSAValue | Operation],
        result_types: List[Attribute],
        before: Region | List[Operation] | List[Block],
        after: Region | List[Operation] | List[Block],
    ) -> While:
        op = While.build(operands=operands,
                         result_types=result_types,
                         regions=[before, after])
        return op


Scf = Dialect([If, For, Yield, Condition, While], [])
