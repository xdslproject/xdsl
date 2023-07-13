from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    VarOperand,
    VarOpResult,
    irdl_op_definition,
    operand_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import HasParent, IsTerminator, SingleBlockImplicitTerminator
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class While(IRDLOperation):
    name = "scf.while"
    arguments: VarOperand = var_operand_def(AnyAttr())

    res: VarOpResult = var_result_def(AnyAttr())
    before_region: Region = region_def()
    after_region: Region = region_def()

    # TODO verify dependencies between scf.condition, scf.yield and the regions
    def verify_(self):
        for idx, arg in enumerate(self.arguments):
            if self.before_region.block.args[idx].type != arg.type:
                raise Exception(
                    f"Block arguments with wrong type, expected {arg.type}, "
                    f"got {self.before_region.block.args[idx].type}"
                )

        for idx, res in enumerate(self.res):
            if self.after_region.block.args[idx].type != res.type:
                raise Exception(
                    f"Block arguments with wrong type, expected {res.type}, "
                    f"got {self.after_region.block.args[idx].type}"
                )

    @staticmethod
    def get(
        operands: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        before: Region | Sequence[Operation] | Sequence[Block],
        after: Region | Sequence[Operation] | Sequence[Block],
    ) -> While:
        return While.build(
            operands=operands, result_types=result_types, regions=[before, after]
        )


@irdl_op_definition
class Yield(IRDLOperation):
    name = "scf.yield"
    arguments: VarOperand = var_operand_def(AnyAttr())

    # TODO circular dependency disallows this set of traits
    # tracked by gh issues https://github.com/xdslproject/xdsl/issues/1218
    # traits = frozenset([HasParent((For, If, ParallelOp, While)), IsTerminator()])
    traits = frozenset([IsTerminator()])

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class If(IRDLOperation):
    name = "scf.if"
    output: VarOpResult = var_result_def(AnyAttr())
    cond: Operand = operand_def(IntegerType(1))

    true_region: Region = region_def()
    # TODO this should be optional under certain conditions
    false_region: Region = region_def()

    traits = frozenset([SingleBlockImplicitTerminator(Yield)])

    @staticmethod
    def get(
        cond: SSAValue | Operation,
        return_types: Sequence[Attribute],
        true_region: Region | Sequence[Block] | Sequence[Operation],
        false_region: Region | Sequence[Block] | Sequence[Operation] | None = None,
    ) -> If:
        if false_region is None:
            false_region = Region()

        return If.build(
            operands=[cond],
            result_types=[return_types],
            regions=[true_region, false_region],
        )


@irdl_op_definition
class For(IRDLOperation):
    name = "scf.for"

    lb: Operand = operand_def(IndexType)
    ub: Operand = operand_def(IndexType)
    step: Operand = operand_def(IndexType)

    iter_args: VarOperand = var_operand_def(AnyAttr())

    res: VarOpResult = var_result_def(AnyAttr())

    body: Region = region_def("single_block")

    traits = frozenset([SingleBlockImplicitTerminator(Yield)])

    def verify_(self):
        if (len(self.iter_args) + 1) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args)+1}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments."
            )
        if self.body.block.args and (iter_var := self.body.block.args[0]):
            if not isinstance(iter_var.type, IndexType):
                raise VerifyException(
                    f"The first block argument of the body is of type {iter_var.type}"
                    " instead of index"
                )
        for idx, arg in enumerate(self.iter_args):
            if self.body.block.args[idx + 1].type != arg.type:
                raise VerifyException(
                    f"Block arguments with wrong type, expected {arg.type}, "
                    f"got {self.body.block.args[idx].type}. Arguments after the "
                    f"induction variable must match the carried variables."
                )
        if len(self.body.ops) > 0 and isinstance(self.body.block.last_op, Yield):
            yieldop = self.body.block.last_op
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The scf.for must yield its carried variables."
                )
            for idx, arg in enumerate(yieldop.arguments):
                if self.iter_args[idx].type != arg.type:
                    raise VerifyException(
                        f"Expected {self.iter_args[idx].type}, got {arg.type}. The "
                        f"scf.for's scf.yield must match carried variables types."
                    )

    @staticmethod
    def get(
        lb: SSAValue | Operation,
        ub: SSAValue | Operation,
        step: SSAValue | Operation,
        iter_args: Sequence[SSAValue | Operation],
        body: Region | Sequence[Operation] | Sequence[Block] | Block,
    ) -> For:
        if isinstance(body, Block):
            body = [body]

        return For.build(
            operands=[lb, ub, step, iter_args],
            result_types=[[SSAValue.get(a).type for a in iter_args]],
            regions=[body],
        )


@irdl_op_definition
class ParallelOp(IRDLOperation):
    name = "scf.parallel"
    lowerBound: VarOperand = var_operand_def(IndexType)
    upperBound: VarOperand = var_operand_def(IndexType)
    step: VarOperand = var_operand_def(IndexType)
    initVals: VarOperand = var_operand_def(AnyAttr())
    res: VarOpResult = var_result_def(AnyAttr())

    body: Region = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments()]

    traits = frozenset([SingleBlockImplicitTerminator(Yield)])

    @staticmethod
    def get(
        lowerBounds: Sequence[SSAValue | Operation],
        upperBounds: Sequence[SSAValue | Operation],
        steps: Sequence[SSAValue | Operation],
        body: Region | Sequence[Block] | Sequence[Operation],
        initVals: Sequence[SSAValue | Operation] = [],
    ) -> ParallelOp:
        return ParallelOp.build(
            operands=[lowerBounds, upperBounds, steps, initVals],
            regions=[body],
            result_types=[[SSAValue.get(a).type for a in initVals]],
        )

    def verify_(self) -> None:
        # This verifies the scf.parallel operation, as can be seen it's fairly complex
        # due to the restrictions on the operation and ability to mix in reduction operations
        # All initvals must be handled by an individual reduction operation, with the block
        # arguments just being induction variables and no arguments to the yield as that is
        # handled by the reduction op

        # First check that the number of lower and upper bounds, along with the number of
        # steps is all equal
        if len(self.lowerBound) != len(self.upperBound) or len(self.lowerBound) != len(
            self.step
        ):
            raise VerifyException(
                "Expected the same number of lower bounds, upper "
                "bounds, and steps for scf.parallel. Got "
                f"{len(self.lowerBound)}, {len(self.upperBound)} and "
                f"{len(self.step)}."
            )

        body_args = self.body.block.args
        # Check the number of block arguments equals the number of induction variables as all
        # initVals must be encapsulated in a reduce operation
        if len(self.lowerBound) != len(body_args):
            raise VerifyException(
                "Number of block arguments must exactly equal the number of induction variables"
            )

        # Check that the number of initial values (initVals)
        # equals the number of reductions
        if len(self.initVals) != self.count_number_reduction_ops():
            raise VerifyException(
                f"Expected {len(self.initVals)} "
                f"reductions but {self.count_number_reduction_ops()} provided"
            )

        # Check each induction variable argument is present in the block arguments
        # and the block argument is of type index
        if not all([isinstance(a.type, IndexType) for a in body_args]):
            raise VerifyException(
                "scf.parallel's block must have an index argument"
                " for each induction variable"
            )

        # Now go through each reduction operation and check that the type
        # matches the corresponding initVals type
        num_reductions = self.count_number_reduction_ops()
        for reduction in range(num_reductions):
            arg_type = self.get_arg_type_of_nth_reduction_op(reduction)
            initValsType = self.initVals[reduction].type
            if initValsType != arg_type:
                raise VerifyException(
                    f"Miss match on scf.parallel argument and reduction op type number {reduction} "
                    f", parallel argment is of type {initValsType} whereas reduction operation is of type {arg_type}"
                )

        # Ensure that the number of operands in scf.yield is exactly zero
        if (last_op := self.body.block.last_op) is not None and last_op.operands:
            raise VerifyException(
                f"Single-block region terminator scf.yield has {len(last_op.operands)} "
                "operands but 0 expected inside an scf.parallel"
            )

        # Ensure that the number of reductions matches the
        # number of result types from scf.parallel
        if num_reductions != len(self.res):
            raise VerifyException(
                f"There are {num_reductions} reductions, but {len(self.res)} results expected"
            )

        # Now go through each reduction and ensure that its operand type matches the corresponding
        # scf.parallel result type (there is no result type on scf.reduce, hence we check the
        # operand type)
        for reduction in range(num_reductions):
            arg_type = self.get_arg_type_of_nth_reduction_op(reduction)
            res_type = self.res[reduction].type
            if res_type != arg_type:
                raise VerifyException(
                    f"Miss match on scf.parallel result type and reduction op type number {reduction} "
                    f", parallel argment is of type {res_type} whereas reduction operation is of type {arg_type}"
                )

    def count_number_reduction_ops(self) -> int:
        num_reduce = 0
        for op in self.body.block.ops:
            if isinstance(op, ReduceOp):
                num_reduce += 1
        return num_reduce

    def get_arg_type_of_nth_reduction_op(self, index: int):
        found = 0
        for op in self.body.block.ops:
            if isinstance(op, ReduceOp):
                if found == index:
                    return op.argument.type
                found += 1
        return None


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "scf.reduce"
    argument: Operand = operand_def(AnyAttr())

    body: Region = region_def("single_block")

    @staticmethod
    def get(
        argument: SSAValue | Operation,
        block: Block,
    ) -> ReduceOp:
        return ReduceOp.build(operands=[argument], regions=[Region(block)])

    def verify_(self) -> None:
        if len(self.body.block.args) != 2:
            raise VerifyException(
                "scf.reduce block must have exactly two arguments, but "
                f"{len(self.body.block.args)} were provided"
            )

        if self.body.block.args[0].type != self.body.block.args[1].type:
            raise VerifyException(
                "scf.reduce block argument types must be the same but have "
                f"{self.body.block.args[0].type} and {self.body.block.args[1].type}"
            )

        if self.body.block.args[0].type != self.argument.type:
            raise VerifyException(
                "scf.reduce block argument types must match the operand type "
                f" but have {self.body.block.args[0].type} and {self.argument.type}"
            )

        last_op = self.body.block.last_op

        if last_op is None or not isinstance(last_op, ReduceReturnOp):
            raise VerifyException(
                "Block inside scf.reduce must terminate with an scf.reduce.return"
            )

        if last_op.result.type != self.argument.type:
            raise VerifyException(
                "scf.reduce.return result type at end of scf.reduce block must"
                f" match the reduction operand type but have {last_op.result.type} "
                f"and {self.argument.type}"
            )


@irdl_op_definition
class ReduceReturnOp(IRDLOperation):
    name = "scf.reduce.return"
    result: Operand = operand_def(AnyAttr())

    traits = frozenset([HasParent(ReduceOp), IsTerminator()])

    @staticmethod
    def get(
        result: SSAValue | Operation,
    ) -> ReduceReturnOp:
        return ReduceReturnOp.build(operands=[result])


@irdl_op_definition
class Condition(IRDLOperation):
    name = "scf.condition"
    cond: Operand = operand_def(IntegerType(1))
    arguments: VarOperand = var_operand_def(AnyAttr())

    traits = frozenset([HasParent(While), IsTerminator()])

    @staticmethod
    def get(cond: SSAValue | Operation, *output_ops: SSAValue | Operation) -> Condition:
        return Condition.build(operands=[cond, [output for output in output_ops]])


Scf = Dialect(
    [
        If,
        For,
        Yield,
        Condition,
        ParallelOp,
        ReduceOp,
        ReduceReturnOp,
        While,
    ],
    [],
)
