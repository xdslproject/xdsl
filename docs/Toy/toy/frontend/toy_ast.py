from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generator, Iterable

from .location import Location

INDENT = 2


@dataclass
class VarType:
    "A variable type with shape information."
    shape: list[int]


class ExprASTKind(Enum):
    Expr_VarDecl = 1
    Expr_Return = 2
    Expr_Num = 3
    Expr_Literal = 4
    Expr_Var = 5
    Expr_BinOp = 6
    Expr_Call = 7
    Expr_Print = 8


@dataclass()
class Dumper:
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


@dataclass
class ExprAST:
    loc: Location

    def __init__(self, loc: Location):
        self.loc = loc
        print(self.dump())

    @property
    def kind(self) -> ExprASTKind:
        raise AssertionError(f"ExprAST kind not defined for {type(self)}")

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, self.__class__.__name__)

    def dump(self):
        dumper = Dumper([])
        self.inner_dump("", dumper)
        return dumper.message


@dataclass
class VarDeclExprAST(ExprAST):
    "Expression class for defining a variable."
    name: str
    varType: VarType
    expr: ExprAST

    @property
    def kind(self):
        return ExprASTKind.Expr_VarDecl

    def inner_dump(self, prefix: str, dumper: Dumper):
        dims_str = ", ".join(f"{int(dim)}" for dim in self.varType.shape)
        dumper.append("VarDecl ", f"{self.name}<{dims_str}> @{self.loc}")
        child = dumper.child()
        self.expr.inner_dump("", child)


@dataclass
class ReturnExprAST(ExprAST):
    "Expression class for a return operator."
    expr: ExprAST | None

    @property
    def kind(self):
        return ExprASTKind.Expr_Return

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, "Return")
        if self.expr is not None:
            child = dumper.child()
            self.expr.inner_dump("", child)


@dataclass
class NumberExprAST(ExprAST):
    'Expression class for numeric literals like "1.0".'
    val: float

    @property
    def kind(self):
        return ExprASTKind.Expr_Num

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, f" {self.val:.6e}")


@dataclass
class LiteralExprAST(ExprAST):
    "Expression class for a literal value."
    values: list[LiteralExprAST | NumberExprAST]
    dims: list[int]

    @property
    def kind(self):
        return ExprASTKind.Expr_Literal

    def __dump(self) -> str:
        dims_str = ", ".join(f"{int(dim)}" for dim in self.dims)
        vals_str = ",".join(
            val.__dump() if isinstance(val, LiteralExprAST) else val.dump()
            for val in self.values
        )
        return f" <{dims_str}>[{vals_str}]"

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append("Literal:", self.__dump() + f" @{self.loc}")

    def iter_flattened_values(self) -> Generator[float, None, None]:
        for value in self.values:
            if isinstance(value, NumberExprAST):
                yield value.val
            else:
                yield from value.iter_flattened_values()

    def flattened_values(self) -> list[float]:
        return list(self.iter_flattened_values())


@dataclass
class VariableExprAST(ExprAST):
    'Expression class for referencing a variable, like "a".'
    name: str

    @property
    def kind(self):
        return ExprASTKind.Expr_Var

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append("var: ", f"{self.name} @{self.loc}")


@dataclass
class BinaryExprAST(ExprAST):
    "Expression class for a binary operator."
    op: str
    lhs: ExprAST
    rhs: ExprAST

    @property
    def kind(self):
        return ExprASTKind.Expr_BinOp

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, f"BinOp: {self.op} @{self.loc}")
        child = dumper.child()
        self.lhs.inner_dump("", child)
        self.rhs.inner_dump("", child)


@dataclass
class CallExprAST(ExprAST):
    "Expression class for function calls."
    callee: str
    args: list[ExprAST]

    @property
    def kind(self):
        return ExprASTKind.Expr_Call

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append_list(
            prefix,
            f"Call '{self.callee}' [ @{self.loc}",
            self.args,
            "]",
            lambda dd, arg: arg.inner_dump("", dd),
        )


@dataclass
class PrintExprAST(ExprAST):
    "Expression class for builtin print calls."
    arg: ExprAST

    @property
    def kind(self):
        return ExprASTKind.Expr_Print

    def inner_dump(self, prefix: str, dumper: Dumper):
        super().inner_dump(prefix, dumper)
        child = dumper.child()
        self.arg.inner_dump("arg: ", child)


@dataclass
class PrototypeAST:
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
        dumper.append("Params: ", f'[{", ".join(arg.name for arg in self.args)}]')


@dataclass
class FunctionAST:
    "This class represents a function definition itself."
    loc: Location
    proto: PrototypeAST
    body: tuple[ExprAST, ...]

    def dump(self):
        dumper = Dumper([])
        self.inner_dump("", dumper)
        return dumper.message

    def inner_dump(self, prefix: str, dumper: Dumper):
        dumper.append(prefix, "Function ")
        child = dumper.child()
        self.proto.inner_dump("proto: ", child)
        child.append_list(
            "Block ",
            "{",
            self.body,
            "} // Block",
            lambda dd, stmt: stmt.inner_dump("", dd),
        )


@dataclass
class ModuleAST:
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
