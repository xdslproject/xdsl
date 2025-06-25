from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from xdsl.dialects.builtin import (
    DenseArrayBase,
    DenseIntElementsAttr,
    IndexType,
    IndexTypeConstr,
    IntegerType,
    SignlessIntegerConstraint,
    StringAttr,
    VectorType,
    i32,
)
from xdsl.ir import Attribute, Block, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Successor,
    VarOperand,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_successor_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import HasCanonicalizationPatternsTrait, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException


class AssertHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.cf import AssertTrue

        return (AssertTrue(),)


@irdl_op_definition
class AssertOp(IRDLOperation):
    """Assert operation with message attribute"""

    name = "cf.assert"

    arg = operand_def(IntegerType(1))
    msg = attr_def(StringAttr)

    traits = traits_def(AssertHasCanonicalizationPatterns())

    def __init__(self, arg: Operation | SSAValue, msg: str | StringAttr):
        if isinstance(msg, str):
            msg = StringAttr(msg)
        super().__init__(
            operands=[arg],
            attributes={"msg": msg},
        )

    assembly_format = "$arg `,` $msg attr-dict"


class BranchOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.cf import (
            SimplifyBrToBlockWithSinglePred,
            SimplifyPassThroughBr,
        )

        return (SimplifyBrToBlockWithSinglePred(), SimplifyPassThroughBr())


@irdl_op_definition
class BranchOp(IRDLOperation):
    """Branch operation"""

    name = "cf.br"

    arguments = var_operand_def()
    successor = successor_def()

    traits = traits_def(IsTerminator(), BranchOpHasCanonicalizationPatterns())

    def __init__(self, dest: Block, *ops: Operation | SSAValue):
        super().__init__(operands=[[op for op in ops]], successors=[dest])

    assembly_format = "$successor (`(` $arguments^ `:` type($arguments) `)`)? attr-dict"


class ConditionalBranchOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.cf import (
            CondBranchTruthPropagation,
            SimplifyCondBranchIdenticalSuccessors,
            SimplifyConstCondBranchPred,
            SimplifyPassThroughCondBranch,
        )

        return (
            SimplifyConstCondBranchPred(),
            SimplifyPassThroughCondBranch(),
            SimplifyCondBranchIdenticalSuccessors(),
            CondBranchTruthPropagation(),
        )


@irdl_op_definition
class ConditionalBranchOp(IRDLOperation):
    """Conditional branch operation"""

    name = "cf.cond_br"

    cond = operand_def(IntegerType(1))
    then_arguments = var_operand_def()
    else_arguments = var_operand_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    then_block = successor_def()
    else_block = successor_def()

    traits = traits_def(
        IsTerminator(), ConditionalBranchOpHasCanonicalizationPatterns()
    )

    def __init__(
        self,
        cond: Operation | SSAValue,
        then_block: Block,
        then_ops: Sequence[Operation | SSAValue],
        else_block: Block,
        else_ops: Sequence[Operation | SSAValue],
    ):
        super().__init__(
            operands=[cond, then_ops, else_ops], successors=[then_block, else_block]
        )

    assembly_format = """
    $cond `,`
    $then_block (`(` $then_arguments^ `:` type($then_arguments) `)`)? `,`
    $else_block (`(` $else_arguments^ `:` type($else_arguments) `)`)?
    attr-dict
    """


class SwitchOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.cf import (
            DropSwitchCasesThatMatchDefault,
            SimplifyConstSwitchValue,
            SimplifyPassThroughSwitch,
            SimplifySwitchFromSwitchOnSameCondition,
            SimplifySwitchWithOnlyDefault,
        )

        return (
            SimplifySwitchWithOnlyDefault(),
            SimplifyConstSwitchValue(),
            SimplifyPassThroughSwitch(),
            DropSwitchCasesThatMatchDefault(),
            SimplifySwitchFromSwitchOnSameCondition(),
        )


@irdl_op_definition
class SwitchOp(IRDLOperation):
    """Switch operation"""

    name = "cf.switch"

    case_values = opt_prop_def(DenseIntElementsAttr)

    flag = operand_def(IndexTypeConstr | SignlessIntegerConstraint)

    default_operands = var_operand_def()

    case_operands = var_operand_def()

    # Copied from AttrSizedSegments
    case_operand_segments = prop_def(DenseArrayBase.constr(i32))

    default_block = successor_def()

    case_blocks = var_successor_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(IsTerminator(), Pure(), SwitchOpHasCanonicalizationPatterns())

    def __init__(
        self,
        flag: Operation | SSAValue,
        default_block: Successor,
        default_operands: Sequence[Operation | SSAValue],
        case_values: DenseIntElementsAttr | None = None,
        case_blocks: Sequence[Successor] = [],
        case_operands: Sequence[Sequence[Operation | SSAValue]] = [],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        case_operand_segments = tuple(len(o) for o in case_operands)
        c_operands: tuple[SSAValue | Operation, ...] = tuple(
            o for os in case_operands for o in os
        )
        operands = [flag, default_operands, c_operands]
        properties: dict[str, Attribute] = {
            "case_operand_segments": DenseArrayBase.from_list(
                i32, case_operand_segments
            )
        }
        if case_values:
            properties["case_values"] = case_values

        successors = [default_block, case_blocks]
        super().__init__(
            operands=operands,
            attributes=attr_dict,
            properties=properties,
            successors=successors,
        )

    @property
    def case_operand(self) -> tuple[VarOperand, ...]:
        if self.case_operand_segments.elt_type != i32:
            raise VerifyException(
                "case_operand_segments is expected to be a DenseArrayBase of i32"
            )

        def_sizes = self.case_operand_segments.get_values()

        if sum(def_sizes) != len(self.case_operands):
            raise VerifyException(
                "Lengths of case operand segment sizes do not sum to the number of case operands"
            )

        cases: list[VarOperand] = []
        prev = 0
        for size in def_sizes:
            cases.append(VarOperand(self.case_operands[prev : prev + size]))
            prev += size

        return tuple(cases)

    # Copying the verification in mlir, does not check that arguments have correct type
    def verify_(self) -> None:
        if not self.case_values:
            if not self.case_blocks:
                return
            raise VerifyException(
                "'Case values' must be specified when there are case blocks"
            )

        if self.flag.type != self.case_values.get_element_type():
            raise VerifyException(
                f"'flag type ({self.flag.type}) should match case value type ({self.case_values.get_element_type()})"
            )

        # Check case values have the correct shape
        shape = self.case_values.get_shape()
        if not shape or len(shape) != 1 or shape[0] != len(self.case_blocks):
            raise VerifyException(
                f"number of case values should match number of case blocks ({len(self.case_blocks)})"
            )

        # Check case operands are well formed
        self.case_operand

    @staticmethod
    def _print_case(
        printer: Printer, case_name: str, block: Block, arguments: VarOperand
    ):
        printer.print_string(case_name)
        printer.print_string(": ")
        printer.print_block_name(block)
        if arguments:
            printer.print_string("(")
            printer.print_list(arguments, printer.print_operand)
            printer.print_string(" : ")
            printer.print_list(arguments.types, printer.print_attribute)
            printer.print_string(")")

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.flag)
        printer.print_string(" : ")
        printer.print_attribute(self.flag.type)
        printer.print_string(", [")
        with printer.indented():
            printer.print_string("\n")
            cases = [("default", self.default_block, self.default_operands)]
            if self.case_values:
                cases = cases + [
                    (str(c), block, operands)
                    for (c, block, operands) in zip(
                        self.case_values.get_values(),
                        self.case_blocks,
                        self.case_operand,
                    )
                ]

            printer.print_list(
                cases, lambda x: self._print_case(printer, x[0], x[1], x[2]), ",\n"
            )

        printer.print_string("\n]")
        attr_dict = {
            k: v
            for k, v in self.attributes.items()
            if k not in ("case_operand_segments", "case_values", "operandSegmentSizes")
        }
        if attr_dict:
            printer.print_attr_dict(attr_dict)

    @staticmethod
    def _parse_case_body(parser: Parser) -> tuple[Block, Sequence[SSAValue]]:
        parser.parse_punctuation(":")
        block = parser.parse_successor()
        if parser.parse_optional_punctuation("("):
            unresolved = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
            parser.parse_punctuation(":")
            types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
            parser.parse_punctuation(")")
            operands = parser.resolve_operands(unresolved, types, parser.pos)
            return (block, operands)
        else:
            return (block, ())

    @classmethod
    def _parse_case(cls, parser: Parser) -> tuple[int, Block, Sequence[SSAValue]]:
        i = parser.parse_integer()
        (block, ops) = cls._parse_case_body(parser)
        return (i, block, ops)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        unresolved_flag = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        flag_type = parser.parse_type()
        flag = parser.resolve_operand(unresolved_flag, flag_type)
        parser.parse_punctuation(",")
        parser.parse_punctuation("[")
        parser.parse_keyword("default")
        (default_block, default_args) = cls._parse_case_body(parser)
        case_values: DenseIntElementsAttr | None = None
        case_blocks: tuple[Block, ...] = ()
        case_operands: tuple[tuple[SSAValue, ...], ...] = ()
        if parser.parse_optional_punctuation(","):
            cases = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, lambda: cls._parse_case(parser)
            )
            assert isinstance(flag_type, IntegerType | IndexType)
            data = tuple(x for (x, _, _) in cases)
            case_values = DenseIntElementsAttr.from_list(
                VectorType(flag_type, (len(data),)), data
            )
            case_blocks = tuple(x for (_, x, _) in cases)
            case_operands = tuple(tuple(x) for (_, _, x) in cases)
        parser.parse_punctuation("]")
        attr_dict = parser.parse_optional_attr_dict()
        return cls(
            flag,
            default_block,
            default_args,
            case_values,
            case_blocks,
            case_operands,
            attr_dict,
        )


Cf = Dialect(
    "cf",
    [
        AssertOp,
        BranchOp,
        ConditionalBranchOp,
        SwitchOp,
    ],
    [],
)
