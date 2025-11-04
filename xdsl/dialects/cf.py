from __future__ import annotations

from collections.abc import Sequence

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
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_successor_def,
)
from xdsl.irdl.declarative_assembly_format import (
    AttributeVariable,
    CustomDirective,
    OperandVariable,
    ParsingState,
    PrintingState,
    SuccessorVariable,
    TypeDirective,
    VariadicOperandVariable,
    VariadicSuccessorVariable,
    irdl_custom_directive,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import HasCanonicalizationPatternsTrait, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


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
    msg = prop_def(StringAttr)

    traits = traits_def(AssertHasCanonicalizationPatterns())

    def __init__(self, arg: Operation | SSAValue, msg: str | StringAttr):
        if isinstance(msg, str):
            msg = StringAttr(msg)
        super().__init__(
            operands=[arg],
            properties={"msg": msg},
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


@irdl_custom_directive
class SwitchOpCases(CustomDirective):
    flag: TypeDirective
    default_block: SuccessorVariable
    default_operands: VariadicOperandVariable
    default_operand_types: TypeDirective
    case_values: AttributeVariable
    case_blocks: VariadicSuccessorVariable
    case_operands: VariadicOperandVariable
    case_operand_segments: AttributeVariable
    case_operand_types: TypeDirective

    @staticmethod
    def _parse_case_body(
        parser: Parser,
    ) -> tuple[Block, Sequence[UnresolvedOperand], Sequence[Attribute]]:
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
            return (block, unresolved, types)
        else:
            return (block, (), ())

    @classmethod
    def _parse_case(
        cls, parser: Parser
    ) -> tuple[int, Block, Sequence[UnresolvedOperand], Sequence[Attribute]]:
        i = parser.parse_integer()
        (block, ops, types) = cls._parse_case_body(parser)
        return (i, block, ops, types)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        parser.parse_keyword("default")
        (default_block, default_operands, default_operand_types) = (
            self._parse_case_body(parser)
        )
        self.default_block.set(state, default_block)
        self.default_operands.set(state, default_operands)
        self.default_operand_types.set(state, default_operand_types)

        if parser.parse_optional_punctuation(","):
            assert isinstance(self.flag.inner, OperandVariable)
            flag_type = state.operand_types[self.flag.inner.index]
            assert flag_type is not None
            flag_type = flag_type[0]
            assert isa(flag_type, IntegerType | IndexType)
            cases = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, lambda: self._parse_case(parser)
            )
            self.case_operand_segments.set(
                state, DenseArrayBase.from_list(i32, tuple(len(x[2]) for x in cases))
            )
            self.case_values.set(
                state,
                DenseIntElementsAttr.from_list(
                    VectorType(flag_type, (len(cases),)), tuple(x[0] for x in cases)
                ),
            )
            self.case_blocks.set(state, tuple(x[1] for x in cases))
            self.case_operands.set(state, tuple(y for x in cases for y in x[2]))
            self.case_operand_types.set(state, tuple(y for x in cases for y in x[3]))
        else:
            self.case_blocks.set_empty(state)
            self.case_operands.set_empty(state)
            self.case_operand_types.set_empty(state)
            self.case_operand_segments.set(state, DenseArrayBase.from_list(i32, ()))
        return True

    @staticmethod
    def _print_case(
        printer: Printer,
        case_name: str | int,
        block: Block,
        arguments: Sequence[SSAValue],
        types: Sequence[Attribute],
    ):
        if isinstance(case_name, str):
            printer.print_string(case_name)
        else:
            printer.print_int(case_name)
        printer.print_string(": ")
        printer.print_block_name(block)
        if arguments:
            with printer.in_parens():
                printer.print_list(arguments, printer.print_operand)
                printer.print_string(" : ")
                printer.print_list(types, printer.print_attribute)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        cases: list[
            tuple[str | int, Successor, Sequence[SSAValue], Sequence[Attribute]]
        ] = [
            (
                "default",
                self.default_block.get(op),
                self.default_operands.get(op),
                self.default_operand_types.inner.get_types(op),
            )
        ]
        if (case_values := self.case_values.get(op)) is not None:
            assert isa(case_values, DenseIntElementsAttr)
            case_blocks = self.case_blocks.get(op)
            case_operands = self.case_operands.get(op)
            case_operand_types = self.case_operand_types.get(op)
            segments = self.case_operand_segments.get(op)
            assert segments is not None
            assert DenseArrayBase.constr(i32).verifies(segments)
            idx = 0
            for c, block, segment in zip(
                case_values.get_values(), case_blocks, segments.get_values()
            ):
                cases.append(
                    (
                        c,
                        block,
                        case_operands[idx : idx + segment],
                        case_operand_types[idx : idx + segment],
                    )
                )
                idx += segment

        with printer.indented():
            printer.print_list(
                cases,
                lambda x: self._print_case(printer, x[0], x[1], x[2], x[3]),
                ",\n",
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

    assembly_format = (
        "$flag `:` type($flag) `,` `[` `\\n`"
        "custom<SwitchOpCases>("
        "  ref(type($flag)),"
        "  $default_block,"
        "  $default_operands,"
        "  type($default_operands),"
        "  $case_values,"
        "  $case_blocks,"
        "  $case_operands,"
        "  $case_operand_segments,"  # We do not have variadic of variadic support
        "  type($case_operands)"
        ") `\\n` `]` attr-dict"
    )

    custom_directives = (SwitchOpCases,)

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
        cases: list[VarOperand] = []
        prev = 0
        for size in self.case_operand_segments.get_values():
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

        if sum(self.case_operand_segments.get_values()) != len(self.case_operands):
            raise VerifyException(
                "Lengths of case operand segment sizes do not sum to the number of case operands"
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
