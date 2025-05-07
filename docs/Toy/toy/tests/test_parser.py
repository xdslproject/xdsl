from pathlib import Path

import pytest

from xdsl.utils.exceptions import ParseError

from ..frontend.location import Location
from ..frontend.parser import ToyParser
from ..frontend.toy_ast import (
    BinaryExprAST,
    CallExprAST,
    FunctionAST,
    LiteralExprAST,
    ModuleAST,
    NumberExprAST,
    PrintExprAST,
    PrototypeAST,
    ReturnExprAST,
    VarDeclExprAST,
    VariableExprAST,
    VarType,
)


def test_parse_ast():
    ast_toy = Path("docs/Toy/examples/ast.toy")

    with open(ast_toy) as f:
        parser = ToyParser(ast_toy, f.read())

    parsed_module_ast = parser.parseModule()

    def loc(line: int, col: int) -> Location:
        return Location(str(ast_toy), line, col)

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
    parser = ToyParser(Path(), program)
    with pytest.raises(ParseError, match="Expected expression"):
        parser.parseIdentifierExpr()


def test_parse_scalar():
    ast_toy = Path("docs/Toy/examples/scalar.toy")

    with open(ast_toy) as f:
        parser = ToyParser(ast_toy, f.read())

    parsed_module_ast = parser.parseModule()

    def loc(line: int, col: int) -> Location:
        return Location(str(ast_toy), line, col)

    module_ast = ModuleAST(
        (
            FunctionAST(
                loc(3, 1),
                PrototypeAST(loc(3, 1), "main", []),
                (
                    VarDeclExprAST(
                        loc(4, 3),
                        "a",
                        VarType([2, 2]),
                        NumberExprAST(loc(4, 17), 5.5),
                    ),
                    PrintExprAST(
                        loc(5, 3),
                        VariableExprAST(loc(5, 9), "a"),
                    ),
                ),
            ),
        )
    )

    assert parsed_module_ast.dump() == module_ast.dump()

    assert parsed_module_ast == module_ast
