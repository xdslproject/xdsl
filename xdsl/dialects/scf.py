from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    IntegerType,
    SignlessIntegerConstraint,
    i64,
)
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_for_op_like,
    print_for_op_like,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
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
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    HasParent,
    IsTerminator,
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class WhileOp(IRDLOperation):
    name = "scf.while"
    arguments = var_operand_def()

    res = var_result_def()
    before_region = region_def()
    after_region = region_def()

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
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "scf.yield"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(ForOp, IfOp, WhileOp, IndexSwitchOp),
            Pure(),
        )
    )


@irdl_op_definition
class IfOp(IRDLOperation):
    name = "scf.if"
    output = var_result_def()
    cond = operand_def(IntegerType(1))

    true_region = region_def("single_block")
    # TODO this should be optional under certain conditions
    false_region = region_def()

    traits = traits_def(
        SingleBlockImplicitTerminator(YieldOp),
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
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
        cond = parser.parse_operand()
        return_types = []
        if parser.parse_optional_punctuation("->"):
            return_types = parser.parse_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )
        else:
            return_types = []

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


class ForOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.scf import (
            RehoistConstInLoops,
            SimplifyTrivialLoops,
        )

        return (SimplifyTrivialLoops(), RehoistConstInLoops())


@irdl_op_definition
class ForOp(IRDLOperation):
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

    def verify_(self):
        # body block verification
        if not self.body.block.args:
            raise VerifyException(
                "Body block must have induction var as first block arg"
            )

        indvar, *block_iter_args = self.body.block.args
        block_iter_args_num = len(block_iter_args)
        iter_args = self.iter_args
        iter_args_num = len(self.iter_args)

        for opnd in (self.lb, self.ub, self.step):
            if opnd.type != indvar.type:
                raise VerifyException(
                    "Expected induction var to be same type as bounds and step"
                )
        if iter_args_num + 1 != block_iter_args_num + 1:
            raise VerifyException(
                f"Expected {iter_args_num + 1} args, but got {block_iter_args_num + 1}. "
                "Body block must have induction and loop-carried variables as args."
            )
        for i, arg in enumerate(iter_args):
            if block_iter_args[i].type != arg.type:
                raise VerifyException(
                    f"Block arg #{i + 1} expected to be {arg.type}, but got {block_iter_args[i].type}. "
                    "Block args after the induction variable must match the loop-carried variables."
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
        print_for_op_like(
            printer,
            self.lb,
            self.ub,
            self.step,
            self.iter_args,
            self.body,
            IndexType,
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        lb, ub, step, iter_arg_operands, body = parse_for_op_like(parser, IndexType())
        _, *iter_args = body.block.args

        for_op = cls(lb, ub, step, iter_arg_operands, body)

        if not iter_args:
            for trait in for_op.get_traits_of_type(SingleBlockImplicitTerminator):
                ensure_terminator(for_op, trait)

        return for_op


@irdl_op_definition
class ParallelOp(IRDLOperation):
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
class ReduceOp(IRDLOperation):
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
class ConditionOp(IRDLOperation):
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


@irdl_op_definition
class IndexSwitchOp(IRDLOperation):
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

    def _verify_region(self, region: Region, name: str):
        yield_op = region.block.last_op
        assert isinstance(yield_op, YieldOp)

        if yield_op.operand_types != self.result_types:
            raise VerifyException(
                f"region {name} returns values of types ({', '.join(str(x) for x in yield_op.operand_types)})"
                f" but expected ({', '.join(str(x) for x in self.result_types)})"
            )

    def verify_(self) -> None:
        if len(self.cases) != len(self.case_regions):
            raise VerifyException(
                f"has {len(self.case_regions)} case regions but {len(self.cases)} case values"
            )

        cases = self.cases
        if len(set(cases.iter_values())) != len(cases):
            raise VerifyException("has duplicate case value")

        self._verify_region(self.default_region, "default")
        for name, region in zip(cases.iter_values(), self.case_regions, strict=True):
            self._verify_region(region, str(name))

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
        arg = parser.parse_operand()
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
        ConditionOp,
        ParallelOp,
        ReduceOp,
        ReduceReturnOp,
        WhileOp,
        IndexSwitchOp,
    ],
    [],
)
