from pathlib import Path

import pytest

from ..parser import Parser


def test_memreftype():
    ast_toy = Path() / 'docs' / 'Toy' / 'examples' / 'ast.toy'

    with open(ast_toy, 'r') as f:
        parser = Parser(ast_toy, f.read())

    parsed_module_ast = parser.parseModule()

    assert parsed_module_ast is not None
