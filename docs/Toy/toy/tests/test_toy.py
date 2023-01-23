from pathlib import Path

from ..parser import Parser


def test_parse_ast():
    ast_toy = Path() / 'docs' / 'Toy' / 'examples' / 'ast.toy'

    with open(ast_toy, 'r') as f:
        parser = Parser(ast_toy, f.read())

    parsed_module_ast = parser.parseModule()

    assert parsed_module_ast is not None
