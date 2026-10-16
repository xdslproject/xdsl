"""
Imperative matching with implicit failure and deferred rewrite candidates.

Based on Jules Merckx's Match dialect at llvm-project revision
e9b3a5db2e3b574d4e6ff69d213eb2305cbe3002:
https://github.com/jumerckx/llvm-project/tree/e9b3a5db2e3b574d4e6ff69d213eb2305cbe3002/mlir/include/mlir/Dialect/Match/IR

This is an incremental port. Unlike MLIR, xDSL does not elide the dialect prefix
of operations inside a matcher.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    i16,
)
from xdsl.dialects.pdl import AnyPDLType, AnyPDLTypeConstr, OperationType, RangeType
from xdsl.ir import Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AtLeast,
    IRDLOperation,
    base,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasAncestor,
    IsolatedFromAbove,
    NoTerminator,
    OptionalSymbolOpInterface,
    SymbolTable,
    SymbolUserOpInterface,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class MatcherOp(IRDLOperation):
    """An ordered matcher with one root operation and at least one success."""

    name = "match.matcher"

    sym_name = opt_prop_def(SymbolNameConstraint())
    sym_visibility = opt_prop_def(StringAttr)
    body = region_def("single_block")

    traits = traits_def(
        IsolatedFromAbove(), NoTerminator(), OptionalSymbolOpInterface()
    )

    def __init__(
        self,
        body: Region | None = None,
        sym_name: str | StringAttr | None = None,
        sym_visibility: str | StringAttr | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(sym_visibility, str):
            sym_visibility = StringAttr(sym_visibility)
        if body is None:
            body = Region(Block(arg_types=[OperationType()]))
        super().__init__(
            properties={"sym_name": sym_name, "sym_visibility": sym_visibility},
            regions=[body],
        )

    def verify_(self) -> None:
        if len(self.body.block.args) != 1:
            raise VerifyException("expected body block to have exactly one argument")
        if self.body.block.args[0].type != OperationType():
            raise VerifyException(
                "expected body block argument to be of type !pdl.operation"
            )
        if not any(isinstance(op, SuccessOp) for op in self.body.walk()):
            raise VerifyException(
                "expected matcher to contain at least one `match.success` "
                "(directly or transitively)"
            )

    @classmethod
    def parse(cls, parser: Parser) -> "MatcherOp":
        visibility = parser.parse_optional_str_literal()
        name = parser.parse_optional_symbol_name()
        parser.parse_keyword("root")
        parser.parse_punctuation("(")
        root = parser.parse_argument()
        parser.parse_punctuation(")")
        body = parser.parse_region([root])
        attributes = parser.parse_optional_attr_dict_with_keyword(
            ("sym_name", "sym_visibility")
        )
        op = cls(body, name, visibility)
        if attributes is not None:
            op.attributes.update(attributes.data)
        return op

    def print(self, printer: Printer) -> None:
        if self.sym_visibility is not None:
            printer.print_string(" ")
            printer.print_attribute(self.sym_visibility)
        if self.sym_name is not None:
            printer.print_string(" ")
            printer.print_symbol_name(self.sym_name.data)
        printer.print_string(" root(")
        if (block := self.body.blocks.first) is not None and block.args:
            printer.print_block_argument(block.args[0])
        printer.print_string(") ")
        printer.print_region(
            self.body, print_entry_block_args=False, print_empty_block=False
        )
        printer.print_op_attributes(self.attributes, print_keyword=True)


class SuccessOpSymbolUserOpInterface(SymbolUserOpInterface):
    def verify(self, op: Operation) -> None:
        assert isinstance(op, SuccessOp)
        if SymbolTable.lookup_symbol(op, op.rewriter) is None:
            raise VerifyException(
                f"references an unknown rewriter symbol: {op.rewriter}"
            )


@irdl_op_definition
class SuccessOp(IRDLOperation):
    """Record a candidate rewrite and its captures, without terminating matching."""

    name = "match.success"

    rewriter = prop_def(SymbolRefAttr)
    benefit = prop_def(IntegerAttr.constr(i16, value=AtLeast(0)))
    inputs = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))

    traits = traits_def(HasAncestor(MatcherOp), SuccessOpSymbolUserOpInterface())

    assembly_format = (
        "$rewriter `benefit` `(` $benefit `)` "
        "(`(` $inputs^ `:` type($inputs) `)`)? attr-dict"
    )

    def __init__(
        self,
        rewriter: str | SymbolRefAttr,
        benefit: int | IntegerAttr,
        inputs: Sequence[SSAValue] = (),
    ):
        if isinstance(rewriter, str):
            rewriter = SymbolRefAttr(rewriter)
        if isinstance(benefit, int):
            benefit = IntegerAttr(benefit, i16)
        super().__init__(
            operands=[inputs], properties={"rewriter": rewriter, "benefit": benefit}
        )


@irdl_op_definition
class HasNameOp(IRDLOperation):
    """Fail the enclosing scope if the operation has a different name."""

    name = "match.has_name"

    input_op = operand_def(OperationType)
    operation_name = prop_def(StringAttr, prop_name="name")

    assembly_format = "$input_op `,` $name attr-dict"

    def __init__(self, input_op: SSAValue, name: str | StringAttr):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(operands=[input_op], properties={"name": name})


Match = Dialect("match", [MatcherOp, SuccessOp, HasNameOp], [])
