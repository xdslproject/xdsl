from collections.abc import Sequence
from typing import Any

from xdsl.parser import Parser

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
        with open(self.path) as file:
            Parser()

            content = file.read()
            objects = parse_custom_file(content)
            for name, obj in objects.items():
                setattr(module, name, obj)


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
