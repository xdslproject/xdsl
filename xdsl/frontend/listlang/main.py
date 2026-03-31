import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import TypeVar

import xdsl.frontend.listlang.list_dialect as list_dialect
import xdsl.frontend.listlang.lowerings as lowerings
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, scf
from xdsl.frontend.listlang import transforms
from xdsl.frontend.listlang.lang_types import (
    ListLangBool,
    ListLangInt,
    ListLangList,
    ListLangType,
    TypedExpression,
)
from xdsl.frontend.listlang.source import CodeCursor, Located, Location, ParseError
from xdsl.ir import Block, SSAValue
from xdsl.printer import Printer
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

RESERVED_KEYWORDS = ["let", "if", "else", "true", "false"]

IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
INTEGER = re.compile(r"[0-9]+")


@dataclass
class Punctuation:
    rg: re.Pattern[str]
    name: str


LET = Punctuation(re.compile(r"let"), "'let'")
EQUAL = Punctuation(re.compile(r"="), "equal")
SEMICOLON = Punctuation(re.compile(r";"), "semicolon")
SINGLE_PERIOD = Punctuation(re.compile(r"\.(?!\.)"), "period")
COMMA = Punctuation(re.compile(r","), "comma")

IF = Punctuation(re.compile(r"if"), "'if'")
ELSE = Punctuation(re.compile(r"else"), "'else'")
TRUE = Punctuation(re.compile(r"true"), "'true'")
FALSE = Punctuation(re.compile(r"false"), "'false'")

STAR = Punctuation(re.compile(r"\*"), "star")
PLUS = Punctuation(re.compile(r"\+"), "plus")
RANGE = Punctuation(re.compile(r"\.\."), "range")
PIPE = Punctuation(re.compile(r"\|"), "pipe")


EQUAL_CMP = Punctuation(re.compile(r"=="), "equality comparator")
LT_CMP = Punctuation(re.compile(r"<"), "less than comparator")
GT_CMP = Punctuation(re.compile(r">"), "greater than comparator")
LTE_CMP = Punctuation(re.compile(r"<="), "less than or equal comparator")
GTE_CMP = Punctuation(re.compile(r">="), "greater than or equal comparator")
NEQ_CMP = Punctuation(re.compile(r"!="), "not equal comparator")

BOOL_AND = Punctuation(re.compile(r"&&"), "boolean and")
BOOL_OR = Punctuation(re.compile(r"\|\|"), "boolean or")
BOOL_NEG = Punctuation(re.compile(r"!"), "boolean negation")

LPAREN = Punctuation(re.compile(r"\("), "left parenthesis")
RPAREN = Punctuation(re.compile(r"\)"), "right parenthesis")
LCURL = Punctuation(re.compile(r"\{"), "left curly bracket")
RCURL = Punctuation(re.compile(r"\}"), "right curly bracket")


@dataclass
class Binding:
    value: SSAValue
    typ: ListLangType


class ParsingContext:
    cursor: CodeCursor
    bindings: ScopedDict[str, Binding]

    def __init__(self, code: str):
        self.cursor = CodeCursor(code)
        self.bindings = ScopedDict()

    def error(self, msg: str) -> ParseError:
        return ParseError(self.cursor.pos, msg)


## Utils


def parse_opt_punct(ctx: ParsingContext, punct: Punctuation) -> Located[bool]:
    """
    Returns True if the punctuation was successfully parsed.
    """
    matched = ctx.cursor.next_regex(punct.rg)
    return Located(matched.loc, matched.value is not None)


def parse_punct(ctx: ParsingContext, punct: Punctuation) -> Location:
    if not (located := parse_opt_punct(ctx, punct)):
        raise ctx.error(f"expected {punct.name}")
    return located.loc


def parse_opt_identifier(ctx: ParsingContext) -> Located[str | None]:
    matched = ctx.cursor.next_regex(IDENT)
    return Located(
        matched.loc,
        matched.value.group() if matched.value is not None else None,
    )


def parse_identifier(ctx: ParsingContext) -> Located[str]:
    if (ident := parse_opt_identifier(ctx)).value is None:
        raise ctx.error("expected variable identifier")
    return Located(ident.loc, ident.value)


def parse_opt_integer(ctx: ParsingContext) -> Located[int | None]:
    matched = ctx.cursor.next_regex(INTEGER)
    return Located(
        matched.loc,
        int(matched.value.group()) if matched.value is not None else None,
    )


def parse_integer(ctx: ParsingContext) -> Located[int]:
    if (lit := parse_opt_integer(ctx)).value is None:
        raise ctx.error("expected integer constant")
    return Located(lit.loc, lit.value)


def name_hint_without_prefix(value: SSAValue) -> str:
    """
    Get the name hint of a value, removing a possible underscore
    prefix if needed.
    """
    name = value.name_hint
    assert name is not None
    if name.startswith("_"):
        return name[1:]
    return name


def compose_name_hints(*names: SSAValue | str) -> str:
    """
    Create a new name hint by composing multiple strings and name hints.
    The new name hint starts with an underscore, as it is a temporary.
    """
    return "_" + "_".join(
        name if isinstance(name, str) else name_hint_without_prefix(name)
        for name in names
    )


T = TypeVar("T")


# TODO: Drop Python 3.11 support to use proper type parameters.
# Linting is disabled for the time being.
def parse_comma_separated(  # noqa: UP047
    ctx: ParsingContext, p: Callable[[], Located[T | None]]
) -> Sequence[Located[T]]:
    result: list[Located[T]] = []
    while (res := p()).value is not None:
        result.append(Located(res.loc, res.value))
        if not parse_opt_punct(ctx, COMMA).value:
            break
    return result


## Expressions


### Atoms


def _parse_opt_expr_atom(
    ctx: ParsingContext, builder: Builder
) -> Located[TypedExpression | None]:
    # Parse parenthesis expression.
    if parse_opt_punct(ctx, LPAREN):
        expr = parse_expr(ctx, builder)
        parse_punct(ctx, RPAREN)
        return Located(expr.loc, expr.value)

    # Parse block expression.
    if (block := parse_opt_block(ctx, builder)).value is not None:
        if block.value.value is None:
            raise ParseError(
                block.value.loc.pos,
                "expected final expression for block in expression position",
            )
        return Located(block.loc, block.value.value)

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

        if_op.results[0].name_hint = "_if_result"
        return Located(
            if_expr.loc,
            TypedExpression(if_op.results[0], then_block_expr.value.value.typ),
        )

    # Parse integer constant.
    if (lit := parse_opt_integer(ctx)).value is not None:
        val = builder.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(lit.value, ListLangInt().xdsl()))
        )
        val.result.name_hint = f"_c{lit.value}"
        return Located(lit.loc, TypedExpression(val.result, ListLangInt()))

    # Parse false constant.
    if false := parse_opt_punct(ctx, FALSE):
        val = builder.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(0, ListLangBool().xdsl()))
        )
        val.result.name_hint = "_false"
        return Located(false.loc, TypedExpression(val.result, ListLangBool()))

    # Parse true constant.
    if true := parse_opt_punct(ctx, TRUE):
        val = builder.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(1, ListLangBool().xdsl()))
        )
        val.result.name_hint = "_true"
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
        true = builder.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(1, ListLangBool().xdsl()))
        )
        true.result.name_hint = "_true"
        negated = builder.insert_op(arith.XOrIOp(to_negate.value.value, true.result))
        negated.result.name_hint = compose_name_hints("not", to_negate.value.value)
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


### Methods


def _parse_opt_expr_atom_with_methods(
    ctx: ParsingContext, builder: Builder
) -> Located[TypedExpression | None]:
    if (x := _parse_opt_expr_atom(ctx, builder)).value is None:
        return x
    x: Located[TypedExpression | None] = Located(x.loc, x.value)

    while parse_opt_punct(ctx, SINGLE_PERIOD):
        method_name = parse_identifier(ctx)
        parse_punct(ctx, LPAREN)

        if (method := x.value.typ.get_method(method_name.value)) is None:
            raise ParseError(
                method_name.loc.pos,
                f"unknown method '{method_name.value}' for type {x.value.typ}",
            )

        lambda_info = None
        if (args := method.get_lambda_arg_type(x.value.typ)) is not None:
            first_pipe_loc = parse_punct(ctx, PIPE)
            arg_idents = parse_comma_separated(ctx, lambda: parse_opt_identifier(ctx))
            parse_punct(ctx, PIPE)

            if len(arg_idents) != len(args):
                raise ParseError(
                    first_pipe_loc.pos,
                    f"expected {len(args)} arguments in {x.value.typ} method "
                    f"'{method.name}' but got {len(arg_idents)}",
                )

            lambda_block = Block([], arg_types=[x.xdsl() for x in args])
            ctx.bindings = ScopedDict(ctx.bindings)

            for ident, typ, val in zip(arg_idents, args, lambda_block.args):
                if ident.value in RESERVED_KEYWORDS:
                    raise ParseError.from_loc(
                        ident.loc, f"'{ident.value}' is a reserved keyword"
                    )
                if ident.value[0] == "_":
                    raise ParseError.from_loc(
                        ident.loc,
                        "variable names cannot start with an underscore",
                    )
                val.name_hint = ident.value
                ctx.bindings[ident.value] = Binding(val, typ)

            lambda_builder = Builder(InsertPoint.at_start(lambda_block))
            lambda_expr = parse_expr(ctx, lambda_builder)
            lambda_builder.insert_op(list_dialect.YieldOp(lambda_expr.value.value))

            assert ctx.bindings.parent is not None
            ctx.bindings = ctx.bindings.parent

            lambda_info = Located(
                lambda_expr.loc, (lambda_block, lambda_expr.value.typ)
            )

        parse_punct(ctx, RPAREN)

        x = Located(x.loc, method.build(builder, x, lambda_info))
        x.value.value.name_hint = "_" + method.name

    return x


### Binary operators


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
        mul_op.result.name_hint = compose_name_hints(
            lhs.value.value, "times", rhs.value.value
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
        add_op.result.name_hint = compose_name_hints(
            lhs.value.value, "plus", rhs.value.value
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
        list_range.result.name_hint = "_int_list"
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
        cmpi_op.result.name_hint = compose_name_hints(
            lhs.value.value, self.arith_code, rhs.value.value
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
        and_op.result.name_hint = compose_name_hints(
            lhs.value.value, "and", rhs.value.value
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
        or_op.result.name_hint = compose_name_hints(
            lhs.value.value, "or", rhs.value.value
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
        Comparator(NEQ_CMP, "ne"),
    ),
    (
        BoolAnd(),
        BoolOr(),
    ),
)


### Expression parser


def parse_opt_expr(
    ctx: ParsingContext, builder: Builder
) -> Located[TypedExpression | None]:
    def priority_level_parser(level: int) -> Located[TypedExpression | None]:
        if level == 0:
            return _parse_opt_expr_atom_with_methods(ctx, builder)

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


def parse_opt_block(
    ctx: ParsingContext,
    builder: Builder,
) -> Located[Located[TypedExpression | None] | None]:
    """
    Parses a block and returns its trailing expression, if there is one. The
    first location is the start of the block, while the second location
    is the start of where the trailing expression is or would be.

    The scope of bindings within the block is contained, meaning a new scope
    level is added to the binding dictionary when parsing the block.
    """

    if not (lcurl := parse_opt_punct(ctx, LCURL)).value:
        return Located(lcurl.loc, None)
    ctx.bindings = ScopedDict(ctx.bindings)
    res = parse_block_content(ctx, builder)
    parse_punct(ctx, RCURL)
    assert ctx.bindings.parent is not None
    ctx.bindings = ctx.bindings.parent

    return Located(lcurl.loc, res.value)


def parse_block(
    ctx: ParsingContext,
    builder: Builder,
) -> Located[Located[TypedExpression | None]]:
    """
    Parses a block and returns its trailing expression, if there is one. The
    first location is the start of the block, while the second location
    is the start of where the trailing expression is or would be.

    The scope of bindings within the block is contained, meaning a new scope
    level is added to the binding dictionary when parsing the block.
    """

    if (block := parse_opt_block(ctx, builder)).value is None:
        raise ParseError(block.loc.pos, "expected block")
    return Located(block.loc, block.value)


## Program


def parse_program(code: str, builder: Builder):
    """
    Parses a program.
    Builds the operations associated with the program, and prints the
    final expression.
    """
    expr = parse_block_content(ParsingContext(code), builder).value.value
    if expr is None:
        return
    expr.typ.print(builder, expr.value)


def program_to_mlir_module(code: str) -> builtin.ModuleOp:
    module = builtin.ModuleOp([])
    builder = Builder(InsertPoint.at_start(module.body.block))

    parse_program(code, builder)

    return module


def program_to_mlir_string(code: str):
    output = io.StringIO()
    Printer(stream=output).print_op(program_to_mlir_module(code))
    return output.getvalue()


if __name__ == "__main__":
    import argparse
    import sys

    from xdsl.context import Context
    from xdsl.transforms import printf_to_llvm

    arg_parser = argparse.ArgumentParser(
        prog="list-lang",
        description="Parses list-lang programs and transforms them.",
    )

    arg_parser.add_argument(
        "filename",
        nargs="?",
        default=sys.stdin,
        help="file to read (defaults to stdin)",
    )

    arg_parser.add_argument(
        "--to", choices=["tensor", "interp", "mlir"], help="conversion target"
    )

    arg_parser.add_argument("--opt", action=argparse.BooleanOptionalAction)

    args = arg_parser.parse_args()

    code = args.filename.read()
    try:
        module = program_to_mlir_module(code)
    except ParseError as e:
        line, col = e.line_column(code)
        raise ValueError(f"Parse error (line {line}, column {col}): {e.msg}")

    module.verify()

    ctx = Context()

    if args.opt:
        transforms.OptimizeListOps().apply(ctx, module)
        module.verify()

    def lower_down_to(target: str | None):
        if target is None:
            return
        lowerings.LowerListToTensor().apply(ctx, module)
        module.verify()
        if target == "tensor":
            return
        lowerings.WrapModuleInFunc().apply(ctx, module)
        module.verify()
        if target == "interp":
            return
        printf_to_llvm.PrintfToLLVM().apply(ctx, module)
        module.verify()
        if target == "mlir":
            return

    lower_down_to(args.to)

    Printer().print_op(module)
    print()
