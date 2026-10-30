from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

from xdsl.dialects.arith import AddiOp, IntegerOverflowFlag
from xdsl.dialects.builtin import (
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    SignlessIntegerConstraint,
    i1,
    i64,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_assignment,
    print_assignment,
)
from xdsl.interfaces import (
    RegionBranchOpInterface,
    RegionBranchTerminatorOpInterface,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    OpResult,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    VarConstraint,
    base,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    ConstantLike,
    HasCanonicalizationPatternsTrait,
    HasParent,
    InvocationBounds,
    IsTerminator,
    OperandRange,
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    RegionBranchPoint,
    RegionSuccessor,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


def _constant_bool(attr: Attribute | None) -> bool | None:
    """Return the value of `attr` if it is an `i1` constant."""
    if isa(attr, BoolAttr):
        return attr.value.data != 0
    return None


def _constant_int(value: SSAValue) -> int | None:
    """Return the value of `value` if it is defined by an integer constant."""
    attr = ConstantLike.get_constant_value(value)
    if isinstance(attr, IntegerAttr):
        return attr.value.data
    return None


@irdl_op_definition
class WhileOp(IRDLOperation, RegionBranchOpInterface):
    name = "scf.while"
    arguments = var_operand_def()

    res = var_result_def()
    before_region = region_def("single_block")
    after_region = region_def("single_block")

    traits = traits_def(RecursiveMemoryEffect())

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

    def entry_successor_operands(self, successor: RegionSuccessor) -> OperandRange:
        return OperandRange(self)

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if (terminator := point.terminator) is None:
            return [RegionSuccessor(self.before_region)]
        # The after region always branches back to the before region.
        if terminator.parent_region() is self.after_region:
            return [RegionSuccessor(self.before_region)]
        return [RegionSuccessor(self), RegionSuccessor(self.after_region)]

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        if successor.is_operation():
            return self.results
        if successor.region is self.before_region:
            return self.before_region.block.args
        return self.after_region.block.args

    # TODO verify that the before region ends with scf.condition and the after region
    # with scf.yield. The types along the edges are verified by the interface.
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
class YieldOp(AbstractYieldOperation[Attribute], RegionBranchTerminatorOpInterface):
    name = "scf.yield"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(ForOp, IfOp, ExecuteRegionOp, WhileOp, IndexSwitchOp),
            Pure(),
        )
    )


class ExecuteRegionOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.scf import (
            SingleBlockExecuteInliner,
        )

        return (SingleBlockExecuteInliner(),)


@irdl_op_definition
class ExecuteRegionOp(IRDLOperation, RegionBranchOpInterface):
    """
    Operation that executes its region exactly once.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfexecute_region-scfexecuteregionop).
    """

    name = "scf.execute_region"

    outs = var_result_def()
    region = region_def()

    traits = traits_def(
        ExecuteRegionOpHasCanonicalizationPatternsTrait(),
    )

    def __init__(
        self,
        result_types: Sequence[Attribute],
        region: Region,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(result_types=(result_types,), regions=(region,))

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if point.is_parent():
            return [RegionSuccessor(self.region)]
        return [RegionSuccessor(self)]

    def region_invocation_bounds(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[InvocationBounds]:
        return [InvocationBounds(1, 1)]

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        return self.results if successor.is_operation() else ()

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        if parser.parse_optional_punctuation("->"):
            result_types = parser.parse_optional_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )
            if result_types is None:
                result_types = [parser.parse_type()]
        else:
            result_types = []

        region = parser.parse_region()
        attr_dict = parser.parse_optional_attr_dict()

        return cls(result_types, region, attr_dict)

    def print(self, printer: Printer):
        print_block_terminators = False
        if bool(self.outs):
            printer.print_string(" -> (")
            printer.print_list(self.outs.types, printer.print_attribute)
            printer.print_string(")")
            print_block_terminators = True

        printer.print_string(" ")
        printer.print_region(
            self.region,
            print_entry_block_args=False,
            print_block_terminators=print_block_terminators,
        )

        if bool(self.attributes.keys()):
            printer.print_attr_dict(self.attributes)

    def verify_(self) -> None:
        if self.region.first_block is None:
            raise VerifyException(
                "scf.execute_region op region needs to have at least one block"
            )


class IfOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.scf import (
            IfPropagateConstantCondition,
        )

        return (IfPropagateConstantCondition(),)


@irdl_op_definition
class IfOp(IRDLOperation, RegionBranchOpInterface):
    name = "scf.if"
    output = var_result_def()
    cond = operand_def(i1)

    true_region = region_def("single_block")
    # TODO this should be optional under certain conditions
    false_region = region_def()

    traits = traits_def(
        SingleBlockImplicitTerminator(YieldOp),
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        IfOpHasCanonicalizationPatternsTrait(),
    )

    def __init__(
        self,
        cond: SSAValue | Operation,
        return_types: Sequence[Attribute],
        true_region: Region | Sequence[Block] | Sequence[Operation],
        false_region: Region | Sequence[Block] | Sequence[Operation] | None = None,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        if false_region is None:
            false_region = Region()

        super().__init__(
            operands=[cond],
            result_types=[return_types],
            regions=[true_region, false_region],
            attributes=attr_dict,
        )

    def _else_successor(self) -> RegionSuccessor:
        # Without an else region, control continues after the op.
        if self.false_region.blocks:
            return RegionSuccessor(self.false_region)
        return RegionSuccessor(self)

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if not point.is_parent():
            return [RegionSuccessor(self)]
        return [RegionSuccessor(self.true_region), self._else_successor()]

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        return self.results if successor.is_operation() else ()

    def entry_successor_regions(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        cond = _constant_bool(operands[0])
        if cond is None:
            return self.successor_regions(RegionBranchPoint.PARENT)
        return [RegionSuccessor(self.true_region) if cond else self._else_successor()]

    def region_invocation_bounds(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[InvocationBounds]:
        cond = _constant_bool(operands[0])
        if cond is None:
            return [InvocationBounds(0, 1)] * 2
        # MLIR keeps the lower bound at 0 even when the condition is known.
        return [InvocationBounds(0, int(cond)), InvocationBounds(0, int(not cond))]

    @staticmethod
    def parse_region_with_yield(parser: Parser) -> Region:
        region = parser.parse_region()
        block = region.blocks.last
        if block is None:
            block = Block()
            region.add_block(block)
        last_op = block.last_op
        if last_op is not None and last_op.has_trait(IsTerminator):
            return region

        block.add_op(YieldOp())

        return region

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        cond = parser.parse_operand(i1)
        return_types = []
        if parser.parse_optional_punctuation("->"):
            return_types = parser.parse_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )

        then_region = cls.parse_region_with_yield(parser)

        else_region = (
            cls.parse_region_with_yield(parser)
            if parser.parse_optional_keyword("else")
            else Region()
        )

        attr_dict = parser.parse_optional_attr_dict()

        return cls(cond, return_types, then_region, else_region, attr_dict)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.cond)

        print_block_terminators = False
        if bool(self.output):
            printer.print_string(" -> (")
            printer.print_list(self.output.types, printer.print_attribute)
            printer.print_string(")")
            print_block_terminators = True

        printer.print_string(" ")
        printer.print_region(
            self.true_region,
            print_entry_block_args=False,
            print_block_terminators=print_block_terminators,
        )

        if bool(self.false_region.blocks):
            printer.print_string(" else ")
            printer.print_region(
                self.false_region,
                print_entry_block_args=False,
                print_block_terminators=print_block_terminators,
            )

        if bool(self.attributes.keys()):
            printer.print_attr_dict(self.attributes)


def _ub_minus_lb(lb: SSAValue, ub: SSAValue) -> int | None:
    """Return `ub - lb` if `ub` is `arith.addi lb, c` with a constant `c` and `nsw`."""
    if (
        not isinstance(ub, OpResult)
        or not isinstance(add_op := ub.owner, AddiOp)
        or IntegerOverflowFlag.NSW not in add_op.overflow_flags.data
        or add_op.lhs is not lb
    ):
        return None
    return _constant_int(add_op.rhs)


def _static_trip_count(lb: SSAValue, ub: SSAValue, step: SSAValue) -> int | None:
    """
    Return the number of iterations of the loop, or None if it is not known statically.

    Port of MLIR `constantTripCount`. Unlike MLIR, the bounds are always compared as signed
    integers, as `scf.for` has no `unsignedCmp` here, and the bitwidth check is skipped, as the
    bounds and step are verified to have the same type.
    """
    if lb is ub:
        return 0
    lb_value = _constant_int(lb)
    if lb_value == 0 and ub is step:
        return 1

    step_value = _constant_int(step)
    if step_value == 0:
        # Either zero or infinitely many iterations.
        return None

    ub_value = _constant_int(ub)
    if lb_value is not None:
        if ub_value is None:
            return None
        if ub_value <= lb_value:
            return 0
        diff = ub_value - lb_value
    else:
        if ub_value is not None:
            return None
        diff = _ub_minus_lb(lb, ub)
        if diff is None:
            return None

    if diff < 0:
        return 0
    # A negative step may never reach the upper bound.
    if step_value is None or step_value < 0:
        return None
    return -(-diff // step_value)


class ForOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.scf import (
            RehoistConstInLoops,
            SimplifyTrivialLoops,
        )

        return (SimplifyTrivialLoops(), RehoistConstInLoops())


@irdl_op_definition
class ForOp(IRDLOperation, RegionBranchOpInterface):
    name = "scf.for"

    T: ClassVar = VarConstraint("T", base(IndexType) | SignlessIntegerConstraint)

    lb = operand_def(T)
    ub = operand_def(T)
    step = operand_def(T)

    iter_args = var_operand_def()

    res = var_result_def()

    body = region_def("single_block")

    traits = traits_def(
        SingleBlockImplicitTerminator(YieldOp),
        ForOpHasCanonicalizationPatternsTrait(),
        RecursiveMemoryEffect(),
    )

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

    def entry_successor_operands(self, successor: RegionSuccessor) -> OperandRange:
        # `iter_args` come after `lb`, `ub` and `step`.
        return OperandRange(self, 3)

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        trip_count = _static_trip_count(self.lb, self.ub, self.step)
        if point.is_parent() and trip_count is not None:
            return [RegionSuccessor(self.body if trip_count else self)]
        if not point.is_parent() and trip_count == 1:
            return [RegionSuccessor(self)]
        return [RegionSuccessor(self.body), RegionSuccessor(self)]

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        if successor.is_operation():
            return self.results
        # Skip the induction variable.
        return self.body.block.args[1:]

    def verify_(self):
        # body block verification
        if not self.body.block.args:
            raise VerifyException(
                "Body block must have induction var as first block arg"
            )

        indvar = self.body.block.args[0]
        iter_args = self.iter_args
        iter_args_num = len(self.iter_args)

        for opnd in (self.lb, self.ub, self.step):
            if opnd.type != indvar.type:
                raise VerifyException(
                    "Expected induction var to be same type as bounds and step"
                )
        if (last_op := self.body.block.last_op) is not None and isinstance(
            last_op, YieldOp
        ):
            yieldop = last_op
            if len(yieldop.arguments) != iter_args_num:
                raise VerifyException(
                    f"{yieldop.name} expected {iter_args_num} args, but got {len(yieldop.arguments)}. "
                    f"The {self.name} must yield its loop-carried variables."
                )
            for i, arg in enumerate(yieldop.arguments):
                if iter_args[i].type != arg.type:
                    raise VerifyException(
                        f"Expected yield arg #{i} to be {iter_args[i].type}, but got {arg.type}. "
                        f"{yieldop.name} of {self.name} must match loop-carried variable types."
                    )

    def print(self, printer: Printer):
        indvar, *iter_args = self.body.block.args
        printer.print_string(" ")
        printer.print_ssa_value(indvar)
        printer.print_string(" = ")
        printer.print_ssa_value(self.lb)
        printer.print_string(" to ")
        printer.print_ssa_value(self.ub)
        printer.print_string(" step ")
        printer.print_ssa_value(self.step)
        if iter_args:
            printer.print_string(" iter_args(")
            printer.print_list(
                zip(iter_args, self.iter_args),
                lambda pair: print_assignment(printer, *pair),
            )
            printer.print_string(") -> (")
            printer.print_list((arg.type for arg in iter_args), printer.print_attribute)
            printer.print_string(")")
        if not isinstance(indvar.type, IndexType):
            printer.print_string(" : ")
            printer.print_attribute(indvar.type)
        printer.print_string(" ")
        printer.print_region(
            self.body,
            print_entry_block_args=False,
            print_empty_block=False,
            print_block_terminators=bool(self.iter_args),
        )
        printer.print_op_attributes(self.attributes)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        unresolved_indvar = parser.parse_argument(expect_type=False)
        parser.parse_punctuation("=")
        lb = parser.parse_unresolved_operand()
        parser.parse_keyword("to")
        ub = parser.parse_unresolved_operand()
        parser.parse_keyword("step")
        step = parser.parse_unresolved_operand()

        pos = parser.pos
        unresolved_iter_args: list[Parser.UnresolvedArgument] = []
        iter_arg_unresolved_operands: list[UnresolvedOperand] = []
        iter_arg_types: list[Attribute] = []
        if parser.parse_optional_keyword("iter_args"):
            for iter_arg, iter_arg_operand in parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, lambda: parse_assignment(parser)
            ):
                unresolved_iter_args.append(iter_arg)
                iter_arg_unresolved_operands.append(iter_arg_operand)
            parser.parse_punctuation("->")
            iter_arg_types = parser.parse_comma_separated_list(
                Parser.Delimiter.PAREN, parser.parse_attribute
            )

        iter_arg_operands = parser.resolve_operands(
            iter_arg_unresolved_operands, iter_arg_types, pos
        )
        iter_args = [
            arg.resolve(typ) for arg, typ in zip(unresolved_iter_args, iter_arg_types)
        ]
        indvar_type = (
            parser.parse_type()
            if parser.parse_optional_punctuation(":")
            else IndexType()
        )
        resolved_lb = parser.resolve_operand(lb, indvar_type)
        resolved_ub = parser.resolve_operand(ub, indvar_type)
        resolved_step = parser.resolve_operand(step, indvar_type)
        indvar = unresolved_indvar.resolve(indvar_type)
        body = parser.parse_region((indvar, *iter_args))
        attrs = parser.parse_optional_attr_dict()

        for_op = cls(resolved_lb, resolved_ub, resolved_step, iter_arg_operands, body)
        for_op.attributes |= attrs

        if not iter_args:
            for trait in for_op.get_traits_of_type(SingleBlockImplicitTerminator):
                ensure_terminator(for_op, trait)

        return for_op


@irdl_op_definition
class ParallelOp(IRDLOperation, RegionBranchOpInterface):
    name = "scf.parallel"
    lowerBound = var_operand_def(IndexType)
    upperBound = var_operand_def(IndexType)
    step = var_operand_def(IndexType)
    initVals = var_operand_def()
    res = var_result_def()

    body = region_def("single_block")

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(ReduceOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        lower_bounds: Sequence[SSAValue | Operation],
        upper_bounds: Sequence[SSAValue | Operation],
        steps: Sequence[SSAValue | Operation],
        body: Region | Sequence[Block] | Sequence[Operation],
        init_vals: Sequence[SSAValue | Operation] = (),
    ):
        super().__init__(
            operands=[lower_bounds, upper_bounds, steps, init_vals],
            regions=[body],
            result_types=[[SSAValue.get(a).type for a in init_vals]],
        )

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        return [RegionSuccessor(self.body), RegionSuccessor(self)]

    def verify_(self) -> None:
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

        reduce_op = self.body.block.last_op
        # Ensured by trait
        assert isinstance(reduce_op, ReduceOp)

        num_reductions = len(reduce_op.reductions)

        # Check that the number of initial values (initVals)
        # equals the number of reductions
        if len(self.initVals) != num_reductions:
            raise VerifyException(
                f"Expected {len(self.initVals)} "
                f"reductions but {num_reductions} provided"
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
        for reduction in range(num_reductions):
            arg_type = reduce_op.args[reduction].type
            initValsType = self.initVals[reduction].type
            if initValsType != arg_type:
                raise VerifyException(
                    f"Miss match on scf.parallel argument and reduction op type number {reduction} "
                    f", parallel argment is of type {initValsType} whereas reduction operation is of type {arg_type}"
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
            arg_type = reduce_op.args[reduction].type
            res_type = self.res[reduction].type
            if res_type != arg_type:
                raise VerifyException(
                    f"Miss match on scf.parallel result type and reduction op type number {reduction} "
                    f", parallel argment is of type {res_type} whereas reduction operation is of type {arg_type}"
                )


@irdl_op_definition
class ReduceOp(IRDLOperation, RegionBranchTerminatorOpInterface):
    name = "scf.reduce"
    args = var_operand_def()

    reductions = var_region_def("single_block")

    traits = lazy_traits_def(
        lambda: (
            RecursiveMemoryEffect(),
            HasParent(ParallelOp),
            IsTerminator(),
            SingleBlockImplicitTerminator(ReduceReturnOp),
        )
    )

    assembly_format = "(`(` $args^ `:` type($args) `)`)? $reductions attr-dict"

    def __init__(
        self,
        args: Sequence[SSAValue | Operation] = (),
        regions: Sequence[Region] = (),
    ):
        super().__init__(operands=(args,), regions=(regions,))

    def successor_operands(self, successor: RegionSuccessor) -> OperandRange:
        # The reduced values are not forwarded.
        return OperandRange(self, 0, 0)

    def verify_(self) -> None:
        if len(self.args) != len(self.reductions):
            raise VerifyException(
                "scf.reduce must have the same number of arguments and regions"
                f"but got {len(self.args)} arguments and {len(self.reductions)} regions"
            )
        for region, argument in zip(self.reductions, self.args):
            if len(region.block.args) != 2:
                raise VerifyException(
                    "scf.reduce block must have exactly two arguments, but "
                    f"{len(region.block.args)} were provided"
                )

            if region.block.args[0].type != region.block.args[1].type:
                raise VerifyException(
                    "scf.reduce block argument types must be the same but have "
                    f"{region.block.args[0].type} and {region.block.args[1].type}"
                )

            if region.block.args[0].type != argument.type:
                raise VerifyException(
                    "scf.reduce block argument types must match the operand type "
                    f" but have {region.block.args[0].type} and {argument.type}"
                )

            last_op = region.block.last_op

            # Should be checked by traits
            assert isinstance(last_op, ReduceReturnOp)

            if last_op.result.type != argument.type:
                raise VerifyException(
                    "scf.reduce.return result type at end of scf.reduce block must"
                    f" match the reduction operand type but have {last_op.result.type} "
                    f"and {argument.type}"
                )


@irdl_op_definition
class ReduceReturnOp(IRDLOperation):
    name = "scf.reduce.return"
    result = operand_def()

    traits = traits_def(HasParent(ReduceOp), IsTerminator(), Pure())

    assembly_format = "$result attr-dict `:` type($result)"

    def __init__(self, result: SSAValue | Operation):
        super().__init__(operands=[result])


@irdl_op_definition
class ConditionOp(IRDLOperation, RegionBranchTerminatorOpInterface):
    name = "scf.condition"
    condition = operand_def(IntegerType(1))
    args = var_operand_def()

    traits = traits_def(HasParent(WhileOp), IsTerminator(), Pure())

    assembly_format = "`(` $condition `)` attr-dict ($args^ `:` type($args))?"

    def __init__(
        self,
        condition: SSAValue | Operation,
        *args: SSAValue | Operation,
    ):
        super().__init__(operands=(condition, args))

    def successor_operands(self, successor: RegionSuccessor) -> OperandRange:
        # Everything but the condition is forwarded.
        return OperandRange(self, 1)

    def successor_regions(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        while_op = self.parent_op()
        assert isinstance(while_op, WhileOp)
        cond = _constant_bool(operands[0])
        regions: list[RegionSuccessor] = []
        if cond is None or cond:
            regions.append(RegionSuccessor(while_op.after_region))
        if cond is None or not cond:
            regions.append(RegionSuccessor(while_op))
        return regions


@irdl_op_definition
class IndexSwitchOp(IRDLOperation, RegionBranchOpInterface):
    name = "scf.index_switch"

    arg = operand_def(IndexType)
    cases = prop_def(DenseArrayBase.constr(i64))

    output = var_result_def()

    default_region = region_def("single_block")
    case_regions = var_region_def("single_block")

    traits = traits_def(RecursiveMemoryEffect(), SingleBlockImplicitTerminator(YieldOp))

    def __init__(
        self,
        arg: Operation | SSAValue,
        cases: DenseArrayBase,
        default_region: Region,
        case_regions: Sequence[Region],
        result_types: Sequence[Attribute],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        properties = {
            "cases": cases,
        }

        super().__init__(
            operands=(arg,),
            attributes=attr_dict,
            properties=properties,
            regions=(default_region, case_regions),
            result_types=(result_types,),
        )

    def successor_regions(self, point: RegionBranchPoint) -> Sequence[RegionSuccessor]:
        if not point.is_parent():
            return [RegionSuccessor(self)]
        return [RegionSuccessor(region) for region in self.regions]

    def successor_inputs(self, successor: RegionSuccessor) -> Sequence[SSAValue]:
        return self.results if successor.is_operation() else ()

    def _live_region(self, arg: Attribute | None) -> Region | None:
        """The region entered for the constant `arg`, or None if `arg` is not constant."""
        if not isinstance(arg, IntegerAttr):
            return None
        for case_value, case_region in zip(self.cases.iter_values(), self.case_regions):
            if case_value == arg.value.data:
                return case_region
        return self.default_region

    def entry_successor_regions(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[RegionSuccessor]:
        if (live_region := self._live_region(operands[0])) is None:
            return self.successor_regions(RegionBranchPoint.PARENT)
        return [RegionSuccessor(live_region)]

    def region_invocation_bounds(
        self, operands: Sequence[Attribute | None]
    ) -> Sequence[InvocationBounds]:
        # Deviates from MLIR, which indexes the regions as [case0..caseN, default] while the
        # default region comes first, and so marks the wrong region as live for every constant.
        live_region = self._live_region(operands[0])
        return [
            InvocationBounds(0, int(live_region is None or live_region is region))
            for region in self.regions
        ]

    def verify_(self) -> None:
        if len(self.cases) != len(self.case_regions):
            raise VerifyException(
                f"has {len(self.case_regions)} case regions but {len(self.cases)} case values"
            )

        cases = self.cases
        if len(set(cases.iter_values())) != len(cases):
            raise VerifyException("has duplicate case value")

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.arg)
        attr_dict = {k: v for k, v in self.attributes.items() if k != "cases"}
        if attr_dict:
            printer.print_string(" ")
            printer.print_attr_dict(attr_dict)
        if self.result_types:
            printer.print_string(" -> ")
            printer.print_list(self.result_types, printer.print_attribute)
        printer.print_string("\n")
        for case_value, case_region in zip(
            self.cases.iter_values(), self.case_regions, strict=True
        ):
            printer.print_string(f"case {case_value} ")
            printer.print_region(case_region)
            printer.print_string("\n")

        printer.print_string("default ")
        printer.print_region(self.default_region)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        arg = parser.parse_operand(IndexType())
        attr_dict = parser.parse_optional_attr_dict()
        result_types: list[TypeAttribute] = []
        if parser.parse_optional_punctuation("->"):
            types = parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_type, parser.parse_type
            )
            if types is None:
                parser.raise_error("result types not found")
            result_types = types
        case_values: list[int] = []
        case_regions: list[Region] = []
        while parser.parse_optional_keyword("case"):
            case_values.append(parser.parse_integer())
            case_regions.append(parser.parse_region())
        cases = DenseArrayBase.from_list(i64, case_values)
        parser.parse_keyword("default")
        default_region = parser.parse_region()
        return cls(arg, cases, default_region, case_regions, result_types, attr_dict)


Scf = Dialect(
    "scf",
    [
        IfOp,
        ForOp,
        YieldOp,
        ExecuteRegionOp,
        ConditionOp,
        ParallelOp,
        ReduceOp,
        ReduceReturnOp,
        WhileOp,
        IndexSwitchOp,
    ],
    [],
)
