from pathlib import Path

import pytest

from ..parser import Parser, ParseError
from ..toy_ast import (
    ModuleAST,
    FunctionAST,
    PrototypeAST,
    VariableExprAST,
    ReturnExprAST,
    BinaryExprAST,
    CallExprAST,
    VarDeclExprAST,
    VarType,
    LiteralExprAST,
    NumberExprAST,
)
from ..location import Location


def test_parse_ast():
    ast_toy = Path("docs/Toy/examples/ast.toy")

    with open(ast_toy, "r") as f:
        parser = Parser(ast_toy, f.read())

    parsed_module_ast = parser.parseModule()

    def loc(line: int, col: int) -> Location:
        return Location(ast_toy, line, col)

    module_ast = ModuleAST(
        (
            FunctionAST(
                loc(4, 1),
                PrototypeAST(
                    loc(4, 1),
                    "multiply_transpose",
                    [
                        VariableExprAST(loc(4, 24), "a"),
                        VariableExprAST(loc(4, 27), "b"),
                    ],
                ),
                (
                    ReturnExprAST(
                        loc(5, 3),
                        BinaryExprAST(
                            loc(5, 25),
                            "*",
                            CallExprAST(
                                loc(5, 10),
                                "transpose",
                                [VariableExprAST(loc(5, 20), "a")],
                            ),
                            CallExprAST(
                                loc(5, 25),
                                "transpose",
                                [VariableExprAST(loc(5, 35), "b")],
                            ),
                        ),
                    ),
                ),
            ),
            FunctionAST(
                loc(8, 1),
                PrototypeAST(loc(8, 1), "main", []),
                (
                    VarDeclExprAST(
                        loc(11, 3),
                        "a",
                        VarType([]),
                        LiteralExprAST(
                            loc(11, 11),
                            [
                                LiteralExprAST(
                                    loc(11, 12),
                                    [
                                        NumberExprAST(loc(11, 13), 1.0),
                                        NumberExprAST(loc(11, 16), 2.0),
                                        NumberExprAST(loc(11, 19), 3.0),
                                    ],
                                    [3],
                                ),
                                LiteralExprAST(
                                    loc(11, 23),
                                    [
                                        NumberExprAST(loc(11, 24), 4.0),
                                        NumberExprAST(loc(11, 27), 5.0),
                                        NumberExprAST(loc(11, 30), 6.0),
                                    ],
                                    [3],
                                ),
                            ],
                            [2, 3],
                        ),
                    ),
                    VarDeclExprAST(
                        loc(15, 3),
                        "b",
                        VarType([2, 3]),
                        LiteralExprAST(
                            loc(15, 17),
                            [
                                NumberExprAST(loc(15, 18), 1.0),
                                NumberExprAST(loc(15, 21), 2.0),
                                NumberExprAST(loc(15, 24), 3.0),
                                NumberExprAST(loc(15, 27), 4.0),
                                NumberExprAST(loc(15, 30), 5.0),
                                NumberExprAST(loc(15, 33), 6.0),
                            ],
                            [6],
                        ),
                    ),
                    VarDeclExprAST(
                        loc(19, 3),
                        "c",
                        VarType([]),
                        CallExprAST(
                            loc(19, 11),
                            "multiply_transpose",
                            [
                                VariableExprAST(loc(19, 30), "a"),
                                VariableExprAST(loc(19, 33), "b"),
                            ],
                        ),
                    ),
                    VarDeclExprAST(
                        loc(22, 3),
                        "d",
                        VarType([]),
                        CallExprAST(
                            loc(22, 11),
                            "multiply_transpose",
                            [
                                VariableExprAST(loc(22, 30), "b"),
                                VariableExprAST(loc(22, 33), "a"),
                            ],
                        ),
                    ),
                    VarDeclExprAST(
                        loc(25, 3),
                        "e",
                        VarType([]),
                        CallExprAST(
                            loc(25, 11),
                            "multiply_transpose",
                            [
                                VariableExprAST(loc(25, 30), "b"),
                                VariableExprAST(loc(25, 33), "c"),
                            ],
                        ),
                    ),
                    VarDeclExprAST(
                        loc(28, 3),
                        "f",
                        VarType([]),
                        CallExprAST(
                            loc(28, 11),
                            "multiply_transpose",
                            [
                                CallExprAST(
                                    loc(28, 30),
                                    "transpose",
                                    [VariableExprAST(loc(28, 40), "a")],
                                ),
                                VariableExprAST(loc(28, 44), "c"),
                            ],
                        ),
                    ),
                ),
            ),
        )
    )

    assert parsed_module_ast == module_ast


def test_parse_error():
    program = "def("
    parser = Parser(Path(), program)
    with pytest.raises(ParseError):
        parser.parseIdentifierExpr()
