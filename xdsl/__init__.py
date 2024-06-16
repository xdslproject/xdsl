from collections.abc import Sequence
from itertools import chain
from typing import Any

from xdsl.traits import SymbolTable

from . import _version

__version__ = _version.get_versions()["version"]

import importlib.abc
import importlib.util
import os
import sys


class CustomFileLoader(importlib.abc.Loader):
    def __init__(self, module_name: str, path: str):
        self.module_name = module_name
        self.path = path

    def create_module(self, spec: Any):
        return None

    def exec_module(self, module):
        from xdsl.dialects.irdl import DialectOp
        from xdsl.interpreters.irdl import make_dialect
        from xdsl.ir.context import MLContext
        from xdsl.parser import Parser

        # Open the irdl file
        with open(self.path) as file:
            # Parse it
            ctx = MLContext()
            ctx.register_all_dialects()
            irdl_module = Parser(ctx, file.read(), self.path).parse_module()

            # Make it a PyRDL Dialect
            dialect_name = os.path.basename(self.path)[-5]
            dialect_op = SymbolTable.lookup_symbol(irdl_module, dialect_name)
            assert isinstance(dialect_op, DialectOp)
            dialect = make_dialect(dialect_op)

            for obj in chain(dialect.attributes, dialect.operations):
                setattr(module, obj.__name__, obj)
            setattr(module, dialect.name.capitalize(), dialect)


class CustomFileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None, target: Any = None):
        filename = fullname.split(".")[-1] + ".irdl"
        if path is None:
            path = [os.getcwd()]

        for entry in path:
            potential_path = os.path.join(entry, filename)
            if os.path.isfile(potential_path):
                loader = CustomFileLoader(fullname, potential_path)
                return importlib.util.spec_from_file_location(
                    fullname, potential_path, loader=loader
                )
        return None


def parse_custom_file(content: str):
    pass


sys.meta_path.insert(0, CustomFileFinder())
