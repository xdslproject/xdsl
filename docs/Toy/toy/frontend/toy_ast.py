from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple

from xdsl.utils.lexer import Location

INDENT = 2


class VarType(NamedTuple):
    "A variable type with shape information."

    shape: list[int]


class Dumper(NamedTuple):
    lines: list[str]
    indentation: int = 0

    def append(self, prefix: str, line: str):
        self.lines.append(" " * self.indentation * INDENT + prefix + line)

    def append_list(
        self,
        prefix: str,
        open_paren: str,
        exprs: Iterable[ExprAST | FunctionAST],
        close_paren: str,
        block: Callable[[Dumper, ExprAST | FunctionAST], None],
    ):
        self.append(prefix, open_paren)
        child = self.child()
        for expr in exprs:
            block(child, expr)
        self.append("", close_paren)

    def child(self):
        return Dumper(self.lines, self.indentation + 1)

    @property
    def message(self):
        return "\n".join(self.lines)


class VarDeclExprAST(NamedTuple):
    "Expression class for defining a variable."

    loc: Location
    name: str
    varType: VarType
    expr: ExprAST

    def inner_dump(self, prefix: str, dumper: Dumper):
        dims_str = ", ".join(f"{int(dim)}" for dim in self.varType.shape)
        dumper.append("VarDecl ", f"{self.name}<{dims_str}> @{self.loc}")
        child = dumper.child()
        self.expr.inner_dump("", child)


class ReturnExprAST(NamedTuple):
    "Expression class for a return operator."

    loc: Location
    expr: ExprAST | None

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, "Return")
        if self.expr is not None:
            child = dumper.child()
            self.expr.inner_dump("", child)


class NumberExprAST(NamedTuple):
    'Expression class for numeric literals like "1.0".'

    loc: Location
    val: float

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, f" {self.val:.6e}")


class LiteralExprAST(NamedTuple):
    "Expression class for a literal value."

    loc: Location
    values: list[LiteralExprAST | NumberExprAST]
    dims: list[int]

    def __dump(self) -> str:
        dims_str = ", ".join(f"{int(dim)}" for dim in self.dims)
        vals_str = ",".join(
            val.__dump() if isinstance(val, LiteralExprAST) else f" {val.val:.6e}"
            for val in self.values
        )
        return f" <{dims_str}>[{vals_str}]"

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append("Literal:", self.__dump() + f" @{self.loc}")


class VariableExprAST(NamedTuple):
    'Expression class for referencing a variable, like "a".'

    loc: Location
    name: str

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append("var: ", f"{self.name} @{self.loc}")


class BinaryExprAST(NamedTuple):
    "Expression class for a binary operator."

    loc: Location
    op: str
    lhs: ExprAST
    rhs: ExprAST

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, f"BinOp: {self.op} @{self.loc}")
        child = dumper.child()
        self.lhs.inner_dump("", child)
        self.rhs.inner_dump("", child)


class CallExprAST(NamedTuple):
    "Expression class for function calls."

    loc: Location
    callee: str
    args: list[ExprAST]

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append_list(
            prefix,
            f"Call '{self.callee}' [ @{self.loc}",
            self.args,
            "]",
            lambda dd, arg: arg.inner_dump("", dd),
        )


class PrintExprAST(NamedTuple):
    "Expression class for builtin print calls."

    loc: Location
    arg: ExprAST

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, self.__class__.__name__)
        child = dumper.child()
        self.arg.inner_dump("arg: ", child)


class PrototypeAST(NamedTuple):
    """
    This class represents the "prototype" for a function, which captures its
    name, and its argument names (thus implicitly the number of arguments the
    function takes).
    """

    loc: Location
    name: str
    args: list[VariableExprAST]

    def dump(self):
        dumper = Dumper([])
        self.inner_dump("", dumper)
        return dumper.message

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append("", f"Proto '{self.name}' @{self.loc}")
        dumper.append("Params: ", f"[{', '.join(arg.name for arg in self.args)}]")


class FunctionAST(NamedTuple):
    "This class represents a function definition itself."

    loc: Location
    proto: PrototypeAST
    body: tuple[ExprAST, ...]

    def dump(self):
        dumper = Dumper([])
        self.inner_dump("", dumper)
        return dumper.message

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, "Function")
        child = dumper.child()
        self.proto.inner_dump("proto: ", child)
        child.append_list(
            "Block ",
            "{",
            self.body,
            "} // Block",
            lambda dd, stmt: stmt.inner_dump("", dd),
        )


class ModuleAST(NamedTuple):
    "This class represents a list of functions to be processed together"

    funcs: tuple[FunctionAST, ...]

    def dump(self):
        dumper = Dumper([])
        self.inner_dump("", dumper)
        return dumper.message

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append_list(
            prefix, "Module:", self.funcs, "", lambda dd, func: func.inner_dump("", dd)
        )


ExprAST = (
    BinaryExprAST
    | VariableExprAST
    | LiteralExprAST
    | CallExprAST
    | NumberExprAST
    | PrintExprAST
    | VarDeclExprAST
    | ReturnExprAST
)
