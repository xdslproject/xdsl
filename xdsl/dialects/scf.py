from __future__ import annotations

from typing import Annotated, Sequence

from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    Operand,
    SingleBlockRegion,
    VarOperand,
    VarOpResult,
    irdl_op_definition,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class If(IRDLOperation):
    name = "scf.if"
    output: Annotated[VarOpResult, AnyAttr()]
    cond: Annotated[Operand, IntegerType(1)]

    true_region: Region
    # TODO this should be optional under certain conditions
    false_region: Region

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
class Yield(IRDLOperation):
    name = "scf.yield"
    arguments: Annotated[VarOperand, AnyAttr()]

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class Condition(IRDLOperation):
    name = "scf.condition"
    cond: Annotated[Operand, IntegerType(1)]
    arguments: Annotated[VarOperand, AnyAttr()]

    @staticmethod
    def get(cond: SSAValue | Operation, *output_ops: SSAValue | Operation) -> Condition:
        return Condition.build(operands=[cond, [output for output in output_ops]])


@irdl_op_definition
class For(IRDLOperation):
    name = "scf.for"

    lb: Annotated[Operand, IndexType]
    ub: Annotated[Operand, IndexType]
    step: Annotated[Operand, IndexType]

    iter_args: Annotated[VarOperand, AnyAttr()]

    res: Annotated[VarOpResult, AnyAttr()]

    body: SingleBlockRegion

    def verify_(self):
        if (len(self.iter_args) + 1) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args)+1}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments."
            )
        for idx, arg in enumerate(self.iter_args):
            if self.body.block.args[idx + 1].typ != arg.typ:
                raise VerifyException(
                    f"Block arguments with wrong type, expected {arg.typ}, "
                    f"got {self.body.block.args[idx].typ}. Arguments after the "
                    f"induction variable must match the carried variables."
                )
        if len(self.iter_args) > 0:
            if len(self.body.ops) == 0 or not isinstance(
                self.body.block.last_op, Yield
            ):
                raise VerifyException(
                    "The scf.for's body does not end with a scf.yield. A scf.for loop "
                    "with loop-carried variables must yield their values at the end of "
                    "its body."
                )
        if len(self.body.ops) > 0 and isinstance(self.body.block.last_op, Yield):
            yieldop = self.body.block.last_op
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The scf.for must yield its carried variables."
                )
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
        iter_args: Sequence[SSAValue | Operation],
        body: Region | Sequence[Operation] | Sequence[Block] | Block,
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
class ParallelOp(IRDLOperation):
    name = "scf.parallel"
    lowerBound: Annotated[VarOperand, IndexType]
    upperBound: Annotated[VarOperand, IndexType]
    step: Annotated[VarOperand, IndexType]
    initVals: Annotated[VarOperand, AnyAttr()]
    res: Annotated[VarOpResult, AnyAttr()]

    body: SingleBlockRegion

    irdl_options = [AttrSizedOperandSegments()]

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
            result_types=[[SSAValue.get(a).typ for a in initVals]],
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
        if not all([isinstance(a.typ, IndexType) for a in body_args]):
            raise VerifyException(
                f"scf.parallel's block must have an index argument"
                " for each induction variable"
            )

        # Now go through each reduction operation and check that the type
        # matches the corresponding initVals type
        num_reductions = self.count_number_reduction_ops()
        for reduction in range(num_reductions):
            typ = self.get_arg_type_of_nth_reduction_op(reduction)
            initValsType = self.initVals[reduction].typ
            if initValsType != typ:
                raise VerifyException(
                    f"Miss match on scf.parallel argument and reduction op type number {reduction} "
                    f", parallel argment is of type {initValsType} whereas reduction operation is of type {typ}"
                )

        # Ensure that scf.yield is the last operation in the block as this is required
        if len(self.body.block.ops) == 0 or not isinstance(
            self.body.block.last_op, Yield
        ):
            raise VerifyException(
                "scf.parallel region must terminate with an scf.yield"
            )

        # Ensure that the number of operands in scf.yield is exactly zero
        number_yield_ops = len(self.body.block.last_op.arguments)
        if number_yield_ops != 0:
            raise VerifyException(
                f"scf.yield contains {number_yield_ops} operands but this must be zero inside an scf.parallel"
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
            typ = self.get_arg_type_of_nth_reduction_op(reduction)
            resultType = self.res[reduction].typ
            if resultType != typ:
                raise VerifyException(
                    f"Miss match on scf.parallel result type and reduction op type number {reduction} "
                    f", parallel argment is of type {resultType} whereas reduction operation is of type {typ}"
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
                    return op.argument.typ
                found += 1
        return None


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "scf.reduce"
    argument: Annotated[Operand, AnyAttr()]

    body: SingleBlockRegion

    @staticmethod
    def get(
        argument: SSAValue | Operation,
        block: Block,
    ) -> ReduceOp:
        return ReduceOp.build(operands=[argument], regions=[Region(block, parent=None)])

    def verify_(self) -> None:
        if len(self.body.block.args) != 2:
            raise VerifyException(
                "scf.reduce block must have exactly two arguments, but "
                f"{len(self.body.block.args)} were provided"
            )

        if self.body.block.args[0].typ != self.body.block.args[1].typ:
            raise VerifyException(
                "scf.reduce block argument types must be the same but have "
                f"{self.body.block.args[0].typ} and {self.body.block.args[1].typ}"
            )

        if self.body.block.args[0].typ != self.argument.typ:
            raise VerifyException(
                "scf.reduce block argument types must match the operand type "
                f" but have {self.body.block.args[0].typ} and {self.argument.typ}"
            )

        if len(self.body.block.ops) == 0 or not isinstance(
            self.body.block.last_op, ReduceReturnOp
        ):
            raise VerifyException(
                "scf.reduce block must terminate with an scf.reduce.return"
            )

        if self.body.block.last_op.result.typ != self.argument.typ:
            raise VerifyException(
                "Type of scf.reduce.return result at end of scf.reduce block must "
                f" match the reduction operand type but have {self.body.block.last_op.result.typ} "
                f"and {self.argument.typ}"
            )


@irdl_op_definition
class ReduceReturnOp(IRDLOperation):
    name = "scf.reduce.return"
    result: Annotated[Operand, AnyAttr()]

    @staticmethod
    def get(
        result: SSAValue | Operation,
    ) -> ReduceReturnOp:
        return ReduceReturnOp.build(operands=[result])

    def verify_(self) -> None:
        assert isinstance(self.parent, Block)
        assert isinstance(self.parent.parent, Region)
        if not isinstance(self.parent.parent.parent, ReduceOp):
            raise VerifyException(
                "scf.reduce.return can only be part of an scf.reduce operation"
            )
        op_index = self.parent.get_operation_index(self)
        if op_index != len(self.parent.ops) - 1:
            raise VerifyException(
                "scf.reduce.return can only appear at the end of an scf.reduce body"
            )


@irdl_op_definition
class While(IRDLOperation):
    name = "scf.while"
    arguments: Annotated[VarOperand, AnyAttr()]

    res: Annotated[VarOpResult, AnyAttr()]
    before_region: Region
    after_region: Region

    # TODO verify dependencies between scf.condition, scf.yield and the regions
    def verify_(self):
        for idx, arg in enumerate(self.arguments):
            if self.before_region.block.args[idx].typ != arg.typ:
                raise Exception(
                    f"Block arguments with wrong type, expected {arg.typ}, "
                    f"got {self.before_region.block.args[idx].typ}"
                )

        for idx, res in enumerate(self.res):
            if self.after_region.block.args[idx].typ != res.typ:
                raise Exception(
                    f"Block arguments with wrong type, expected {res.typ}, "
                    f"got {self.after_region.block.args[idx].typ}"
                )

    @staticmethod
    def get(
        operands: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        before: Region | Sequence[Operation] | Sequence[Block],
        after: Region | Sequence[Operation] | Sequence[Block],
    ) -> While:
        op = While.build(
            operands=operands, result_types=result_types, regions=[before, after]
        )
        return op


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
