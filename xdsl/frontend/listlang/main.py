import io
import re
from dataclasses import dataclass
from typing import Generic, cast

from typing_extensions import TypeVar

import xdsl.frontend.list_dialect as list_dialect
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.printer import Printer
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

RESERVED_KEYWORDS = ["let", "if", "else", "true", "false"]

IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
INTEGER = re.compile(r"[0-9]+")

COMMENTS = r"(?:\/\/[^\n\r]+?(?:\*\)|[\n\r]))"
WHITESPACES = re.compile(r"(?:\s|" + COMMENTS + r")*")


@dataclass
class Punctuation:
    rg: re.Pattern[str]
    name: str


LET = Punctuation(re.compile(r"let"), "'let'")
EQUAL = Punctuation(re.compile(r"="), "equal")
SEMICOLON = Punctuation(re.compile(r";"), "semicolon")

IF = Punctuation(re.compile(r"if"), "'if'")
ELSE = Punctuation(re.compile(r"else"), "'else'")
TRUE = Punctuation(re.compile(r"true"), "'true'")
FALSE = Punctuation(re.compile(r"false"), "'false'")

STAR = Punctuation(re.compile(r"\*"), "star")
PLUS = Punctuation(re.compile(r"\+"), "plus")
RANGE = Punctuation(re.compile(r"\.\."), "range")

EQUAL_CMP = Punctuation(re.compile(r"=="), "equality comparator")
LT_CMP = Punctuation(re.compile(r"<"), "less than comparator")
GT_CMP = Punctuation(re.compile(r">"), "greater than comparator")
LTE_CMP = Punctuation(re.compile(r"<="), "less than or equal comparator")
GTE_CMP = Punctuation(re.compile(r">="), "greater than or equal comparator")

BOOL_AND = Punctuation(re.compile(r"&&"), "boolean and")
BOOL_OR = Punctuation(re.compile(r"\|\|"), "boolean or")
BOOL_NEG = Punctuation(re.compile(r"!"), "boolean negation")

LPAREN = Punctuation(re.compile(r"\("), "left parenthesis")
RPAREN = Punctuation(re.compile(r"\)"), "right parenthesis")
LCURL = Punctuation(re.compile(r"\{"), "left curly bracket")
RCURL = Punctuation(re.compile(r"\}"), "right curly bracket")


XDSL_INT = builtin.IntegerType(32)
XDSL_BOOL = builtin.IntegerType(1)


@dataclass
class Location:
    pos: int


T = TypeVar("T")


@dataclass
class Located(Generic[T]):  # noqa: UP046
    loc: Location
    value: T

    def __bool__(self) -> bool:
        return bool(self.value)


class ListLangType:
    def __str__(self) -> str: ...

    def xdsl(self) -> Attribute: ...


@dataclass
class ListLangInt(ListLangType):
    def __str__(self) -> str:
        return "int"

    def xdsl(self) -> builtin.IntegerType:
        return XDSL_INT


@dataclass
class ListLangBool(ListLangType):
    def __str__(self) -> str:
        return "bool"

    def xdsl(self) -> builtin.IntegerType:
        return XDSL_BOOL


@dataclass
class ListLangList(ListLangType):
    element_type: ListLangBool | ListLangInt

    def __str__(self) -> str:
        return f"list<{self.element_type}>"

    def xdsl(self) -> list_dialect.ListType:
        return list_dialect.ListType(self.element_type.xdsl())


@dataclass
class Binding:
    value: SSAValue
    typ: ListLangType


class CodeCursor:
    code: str
    pos: int

    def __init__(self, code: str):
        self.code = code
        self.pos = 0

    def _whitespace_end(self) -> int:
        match = WHITESPACES.match(self.code, self.pos)
        assert match is not None
        return match.end()

    def skip_whitespaces(self):
        self.pos = self._whitespace_end()

    def next_regex(self, regex: re.Pattern[str]) -> Located[re.Match[str] | None]:
        match = self.peek_regex(regex)
        if match.value is not None:
            self.pos = match.value.end()
        return match

    def peek_regex(self, regex: re.Pattern[str]) -> Located[re.Match[str] | None]:
        pos = self._whitespace_end()
        return Located(Location(pos), regex.match(self.code, pos))


class ParsingContext:
    cursor: CodeCursor
    bindings: ScopedDict[str, Binding]

    def __init__(self, code: str):
        self.cursor = CodeCursor(code)
        self.bindings = ScopedDict()


@dataclass
class ParseError(Exception):
    position: int
    msg: str

    @staticmethod
    def from_ctx(ctx: ParsingContext, msg: str) -> "ParseError":
        return ParseError(ctx.cursor.pos, msg)

    @staticmethod
    def from_loc(loc: Location, msg: str) -> "ParseError":
        return ParseError(loc.pos, msg)


@dataclass
class TypedExpression:
    value: SSAValue
    typ: ListLangType


## Utils


def parse_opt_punct(ctx: ParsingContext, punct: Punctuation) -> Located[bool]:
    """
    Returns True if the punctuation was successfully parsed.
    """
    matched = ctx.cursor.next_regex(punct.rg)
    return Located(matched.loc, matched.value is not None)


def parse_punct(ctx: ParsingContext, punct: Punctuation) -> Location:
    if not (located := parse_opt_punct(ctx, punct)):
        raise ParseError.from_ctx(ctx, f"expected {punct.name}")
    return located.loc


def parse_opt_identifier(ctx: ParsingContext) -> Located[str | None]:
    matched = ctx.cursor.next_regex(IDENT)
    return Located(
        matched.loc,
        matched.value.group() if matched.value is not None else None,
    )


def parse_identifier(ctx: ParsingContext) -> Located[str]:
    if (ident := parse_opt_identifier(ctx)).value is None:
        raise ParseError.from_ctx(ctx, "expected variable identifier")
    return Located(ident.loc, ident.value)


def parse_opt_integer(ctx: ParsingContext) -> Located[int | None]:
    matched = ctx.cursor.next_regex(INTEGER)
    return Located(
        matched.loc,
        int(matched.value.group()) if matched.value is not None else None,
    )


def parse_integer(ctx: ParsingContext) -> Located[int]:
    if (lit := parse_opt_integer(ctx)).value is None:
        raise ParseError.from_ctx(ctx, "expected integer constant")
    return Located(lit.loc, lit.value)


## Expressions


def _parse_opt_expr_atom(
    ctx: ParsingContext, builder: Builder
) -> Located[TypedExpression | None]:
    # Parse parenthesis expression.
    if parse_opt_punct(ctx, LPAREN):
        expr = parse_expr(ctx, builder)
        parse_punct(ctx, RPAREN)
        return Located(expr.loc, expr.value)

    # Parse if-expr.
    if if_expr := parse_opt_punct(ctx, IF):
        cond = parse_expr(ctx, builder)

        if not isinstance(cond.value.typ, ListLangBool):
            raise ParseError(
                cond.loc.pos,
                f"expected {ListLangBool()} type for condition, got {cond.value.typ}",
            )

        then_block = Block()
        then_builder = Builder(InsertPoint.at_start(then_block))
        then_block_expr = parse_block(ctx, then_builder)
        if then_block_expr.value.value is None:
            raise ParseError(then_block_expr.value.loc.pos, "expected block expression")
        then_builder.insert_op(scf.YieldOp(then_block_expr.value.value.value))

        parse_punct(ctx, ELSE)

        else_block = Block()
        else_builder = Builder(InsertPoint.at_start(else_block))
        else_block_expr = parse_block(ctx, else_builder)
        if else_block_expr.value.value is None:
            raise ParseError(else_block_expr.value.loc.pos, "expected block expression")
        else_builder.insert_op(scf.YieldOp(else_block_expr.value.value.value))

        if then_block_expr.value.value.typ != else_block_expr.value.value.typ:
            raise ParseError(
                else_block_expr.value.loc.pos,
                "else-block expression should be "
                f"of type {then_block_expr.value.value.typ} "
                f"to match then-block, but "
                f"got {else_block_expr.value.value.typ}",
            )

        if_op = builder.insert_op(
            scf.IfOp(
                cond.value.value,
                [then_block_expr.value.value.typ.xdsl()],
                [then_block],
                [else_block],
            )
        )

        if_op.results[0].name_hint = "if_result"
        return Located(
            if_expr.loc,
            TypedExpression(if_op.results[0], then_block_expr.value.value.typ),
        )

    # Parse integer constant.
    if (lit := parse_opt_integer(ctx)).value is not None:
        val = builder.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(lit.value, XDSL_INT))
        )
        val.result.name_hint = f"c{lit.value}"
        return Located(lit.loc, TypedExpression(val.result, ListLangInt()))

    # Parse false constant.
    if false := parse_opt_punct(ctx, FALSE):
        val = builder.insert_op(arith.ConstantOp(builtin.IntegerAttr(0, XDSL_BOOL)))
        val.result.name_hint = "false"
        return Located(false.loc, TypedExpression(val.result, ListLangBool()))

    # Parse true constant.
    if true := parse_opt_punct(ctx, TRUE):
        val = builder.insert_op(arith.ConstantOp(builtin.IntegerAttr(1, XDSL_BOOL)))
        val.result.name_hint = "true"
        return Located(true.loc, TypedExpression(val.result, ListLangBool()))

    # Parse boolean negation.
    if neg := parse_opt_punct(ctx, BOOL_NEG):
        to_negate = _parse_expr_atom(ctx, builder)
        if not isinstance(to_negate.value.typ, ListLangBool):
            raise ParseError(
                to_negate.loc.pos,
                f"expected {ListLangBool()} type for negation, "
                f"got {to_negate.value.typ}",
            )
        true = builder.insert_op(arith.ConstantOp(builtin.IntegerAttr(1, XDSL_BOOL)))
        negated = builder.insert_op(arith.XOrIOp(to_negate.value.value, true.result))
        negated.result.name_hint = f"not_{to_negate.value.value.name_hint}"
        return Located(neg.loc, TypedExpression(negated.result, ListLangBool()))

    # Parse binding.
    if (ident := parse_opt_identifier(ctx)).value is not None:
        if ident.value not in ctx.bindings:
            raise ParseError.from_loc(ident.loc, "unknown variable")
        binding = ctx.bindings[ident.value]
        return Located(ident.loc, TypedExpression(binding.value, binding.typ))

    return Located(Location(ctx.cursor.pos), None)


def _parse_expr_atom(ctx: ParsingContext, builder: Builder) -> Located[TypedExpression]:
    if (expr := _parse_opt_expr_atom(ctx, builder)).value is None:
        raise ParseError(expr.loc.pos, "expected expression atom")
    return Located(expr.loc, expr.value)


class BinaryOp:
    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        """
        Returns true if the expected glyph was found.
        """
        ...

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression: ...


class Multiplication(BinaryOp):
    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        return parse_opt_punct(ctx, STAR)

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression:
        if not isinstance(lhs.value.typ, ListLangInt):
            raise ParseError(
                lhs.loc.pos,
                f"expected expression of type {ListLangInt()} "
                f"in multiplication, got {lhs.value.typ}",
            )

        if not isinstance(rhs.value.typ, ListLangInt):
            raise ParseError(
                rhs.loc.pos,
                f"expected expression of type {ListLangInt()} "
                f"in multiplication, got {rhs.value.typ}",
            )

        mul_op = builder.insert_op(arith.MuliOp(lhs.value.value, rhs.value.value))
        mul_op.result.name_hint = (
            f"{lhs.value.value.name_hint}_times_{rhs.value.value.name_hint}"
        )
        return TypedExpression(mul_op.result, lhs.value.typ)


class Addition(BinaryOp):
    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        return parse_opt_punct(ctx, PLUS)

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression:
        if not isinstance(lhs.value.typ, ListLangInt):
            raise ParseError(
                lhs.loc.pos,
                f"expected expression of type {ListLangInt()} "
                f"in addition, got {lhs.value.typ}",
            )

        if not isinstance(rhs.value.typ, ListLangInt):
            raise ParseError(
                rhs.loc.pos,
                f"expected expression of type {ListLangInt()} "
                f"in addition, got {rhs.value.typ}",
            )

        add_op = builder.insert_op(arith.AddiOp(lhs.value.value, rhs.value.value))
        add_op.result.name_hint = (
            f"{lhs.value.value.name_hint}_plus_{rhs.value.value.name_hint}"
        )
        return TypedExpression(add_op.result, lhs.value.typ)


class ListRange(BinaryOp):
    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        return parse_opt_punct(ctx, RANGE)

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression:
        lower = lhs
        upper = rhs

        if not isinstance(lower.value.typ, ListLangInt):
            raise ParseError(
                lower.loc.pos,
                f"expected {ListLangInt()} type for range lower bound, "
                f"got {lower.value.typ}",
            )

        if not isinstance(upper.value.typ, ListLangInt):
            raise ParseError(
                upper.loc.pos,
                f"expected {ListLangInt()} type for range upper bound, "
                f"got {upper.value.typ}",
            )

        list_type = ListLangList(ListLangInt())
        list_range = builder.insert(
            list_dialect.RangeOp(
                lower.value.value,
                upper.value.value,
                list_type.xdsl(),
            )
        )
        list_range.result.name_hint = "int_list"
        return TypedExpression(list_range.result, list_type)


@dataclass
class Comparator(BinaryOp):
    glyph: Punctuation
    arith_code: str

    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        return parse_opt_punct(ctx, self.glyph)

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression:
        if not isinstance(lhs.value.typ, ListLangInt):
            raise ParseError(
                lhs.loc.pos,
                f"expected expression of type {ListLangInt()} "
                f"in comparison, got {lhs.value.typ}",
            )

        if not isinstance(rhs.value.typ, ListLangInt):
            raise ParseError(
                rhs.loc.pos,
                f"expected expression of type {ListLangInt()} "
                f"in comparison, got {rhs.value.typ}",
            )

        cmpi_op = builder.insert_op(
            arith.CmpiOp(lhs.value.value, rhs.value.value, self.arith_code)
        )
        cmpi_op.result.name_hint = (
            f"{lhs.value.value.name_hint}_{self.arith_code}_{rhs.value.value.name_hint}"
        )
        return TypedExpression(cmpi_op.result, ListLangBool())


class BoolAnd(BinaryOp):
    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        return parse_opt_punct(ctx, BOOL_AND)

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression:
        if not isinstance(lhs.value.typ, ListLangBool):
            raise ParseError(
                lhs.loc.pos,
                f"expected expression of type {ListLangBool()} "
                f"in boolean and, got {lhs.value.typ}",
            )

        if not isinstance(rhs.value.typ, ListLangBool):
            raise ParseError(
                rhs.loc.pos,
                f"expected expression of type {ListLangBool()} "
                f"in boolean and, got {rhs.value.typ}",
            )

        and_op = builder.insert_op(arith.AndIOp(lhs.value.value, rhs.value.value))
        and_op.result.name_hint = (
            f"{lhs.value.value.name_hint}_and_{rhs.value.value.name_hint}"
        )
        return TypedExpression(and_op.result, lhs.value.typ)


class BoolOr(BinaryOp):
    def parse_opt_glyph(self, ctx: ParsingContext) -> Located[bool]:
        return parse_opt_punct(ctx, BOOL_OR)

    def build(
        self,
        builder: Builder,
        lhs: Located[TypedExpression],
        rhs: Located[TypedExpression],
    ) -> TypedExpression:
        if not isinstance(lhs.value.typ, ListLangBool):
            raise ParseError(
                lhs.loc.pos,
                f"expected expression of type {ListLangBool()} "
                f"in boolean or, got {lhs.value.typ}",
            )

        if not isinstance(rhs.value.typ, ListLangBool):
            raise ParseError(
                rhs.loc.pos,
                f"expected expression of type {ListLangBool()} "
                f"in boolean or, got {rhs.value.typ}",
            )

        or_op = builder.insert_op(arith.OrIOp(lhs.value.value, rhs.value.value))
        or_op.result.name_hint = (
            f"{lhs.value.value.name_hint}_or_{rhs.value.value.name_hint}"
        )
        return TypedExpression(or_op.result, lhs.value.typ)


PARSE_BINOP_PRIORITY: tuple[tuple[BinaryOp, ...], ...] = (
    (Multiplication(),),
    (Addition(),),
    (
        ListRange(),
        Comparator(EQUAL_CMP, "eq"),
        Comparator(LTE_CMP, "ule"),
        Comparator(GTE_CMP, "uge"),
        Comparator(GT_CMP, "ugt"),
        Comparator(LT_CMP, "ult"),
    ),
    (
        BoolAnd(),
        BoolOr(),
    ),
)


def parse_opt_expr(
    ctx: ParsingContext, builder: Builder
) -> Located[TypedExpression | None]:
    def priority_level_parser(level: int) -> Located[TypedExpression | None]:
        if level == 0:
            return _parse_opt_expr_atom(ctx, builder)

        if (lhs := priority_level_parser(level - 1)).value is None:
            return lhs

        lhs = Located(lhs.loc, lhs.value)

        def parse_next_operator_glyph() -> BinaryOp | None:
            operators = PARSE_BINOP_PRIORITY[level - 1]
            return next((op for op in operators if op.parse_opt_glyph(ctx)), None)

        while (selected_op := parse_next_operator_glyph()) is not None:
            if (rhs := priority_level_parser(level - 1)).value is None:
                raise ParseError(rhs.loc.pos, "expected expression")

            expr = selected_op.build(
                builder,
                Located(lhs.loc, lhs.value),
                Located(rhs.loc, rhs.value),
            )
            lhs = Located(lhs.loc, expr)

        return cast(Located[TypedExpression | None], lhs)

    return priority_level_parser(len(PARSE_BINOP_PRIORITY))


def parse_expr(ctx: ParsingContext, builder: Builder) -> Located[TypedExpression]:
    if (expr := parse_opt_expr(ctx, builder)).value is None:
        raise ParseError(expr.loc.pos, "expected expression")
    return Located(expr.loc, expr.value)


## Statements


def parse_opt_let_statement(ctx: ParsingContext, builder: Builder) -> Located[bool]:
    """
    Parses a let statement and adds its binding to the provided context if it
    is there. Returns True if a binding was found, False otherwise.
    """

    if not (let := parse_opt_punct(ctx, LET)):
        return let

    binding_name = parse_identifier(ctx)

    if binding_name.value in RESERVED_KEYWORDS:
        raise ParseError.from_loc(
            binding_name.loc, f"'{binding_name.value}' is a reserved keyword"
        )

    parse_punct(ctx, EQUAL)

    expr = parse_expr(ctx, builder)

    parse_punct(ctx, SEMICOLON)

    expr.value.value.name_hint = binding_name.value

    ctx.bindings[binding_name.value] = Binding(expr.value.value, expr.value.typ)

    return let


def parse_opt_statement(ctx: ParsingContext, builder: Builder) -> Located[bool]:
    return parse_opt_let_statement(ctx, builder)


## Blocks


def parse_block_content(
    ctx: ParsingContext, builder: Builder
) -> Located[Located[TypedExpression | None]]:
    """
    Parses the content of a block and returns its trailing expression, if there
    is one. The first location is the start of the block content, while the
    second location is the start of where the trailing expression is or would
    be.
    """

    ctx.cursor.skip_whitespaces()
    start_loc = Location(ctx.cursor.pos)

    while parse_opt_statement(ctx, builder).value:
        pass

    return Located(start_loc, parse_opt_expr(ctx, builder))


def parse_block(
    ctx: ParsingContext, builder: Builder
) -> Located[Located[TypedExpression | None]]:
    """
    Parses a block and returns its trailing expression, if there is one. The
    first location is the start of the block content, while the second location
    is the start of where the trailing expression is or would be.

    The scope of bindings within the block is contained, meaning a new scope
    level is added to the binding dictionary when parsing the block.
    """

    ctx.bindings = ScopedDict(ctx.bindings)
    parse_punct(ctx, LCURL)
    res = parse_block_content(ctx, builder)
    parse_punct(ctx, RCURL)
    assert ctx.bindings.parent is not None
    ctx.bindings = ctx.bindings.parent

    return res


## Program


def parse_program(code: str, builder: Builder) -> Located[TypedExpression | None]:
    """
    Parses a program.
    If the program has a result expression, returns it. The location of the
    return value is where the type expression is or would be.
    """

    return parse_block_content(ParsingContext(code), builder).value


def program_to_mlir(code: str) -> str:
    module = builtin.ModuleOp([])
    builder = Builder(InsertPoint.at_start(module.body.block))

    parse_program(code, builder)

    output = io.StringIO()
    Printer(stream=output).print_op(module)
    return output.getvalue()


if __name__ == "__main__":
    import fileinput

    program = "\n".join(fileinput.input())

    output = program_to_mlir(program)

    print(output)
