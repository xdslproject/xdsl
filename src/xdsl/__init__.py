import sys
from importlib.abc import MetaPathFinder as _xdsl_importlib_MetaPathFinder
from importlib import util as _xdsl_importlib_util

# Name of the mlir Python module that should be loaded as alias for `mlir_xdsl`.
mlir_module_name = 'mlir'


def _register_mlir_finder():
    # This class acts as a 'finder' in the module find_spec phase of Python's
    # import process. It only handles the case where the module to be imported
    # is called `mlir_xdsl` and returns None otherwise (which means that the
    # next finder is tried). In the former case, it uses the default find_spec
    # implementation to find the module with the name defined in
    # mlir_module_name, effectively aliasing that module as `mlir_xdsl`.
    # See also:
    #   - https://www.sobyte.net/post/2021-10/python-import/
    #   - https://docs.python.org/3/library/importlib.html#importlib.util.find_spec
    class MlirModuleFinder(_xdsl_importlib_MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'mlir_xdsl':
                # Call default find_spec implementation with `mlir_module_name`
                spec = _xdsl_importlib_util.find_spec(mlir_module_name)
                # Override the module name to be sure it is loaded under the
                # alias
                spec.name = 'mlir_xdsl'
                return spec

    # Register our finder
    sys.meta_path += [MlirModuleFinder()]


_register_mlir_finder()
