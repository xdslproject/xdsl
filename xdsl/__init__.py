_mlir_module = None
_REQUIRED_MLIR_MODULES = ['ir']

import types as _xdsl_init_types

# pyright: reportMissingImports=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false


def load_mlir_module(mlir_module: _xdsl_init_types.ModuleType) -> None:
    """
    Configures xDSL to use the given MLIR module. This allows to use xDSL with
    out-of-tree dialects, which ship with their own copy of the MLIR core
    modules (see https://github.com/xdslproject/xdsl/issues/111). If so desired,
    the user needs to call this function prior to importing `mlir_converter`.
    Otherwise, `mlir` is used. Subsequent calls to this function are only
    allowed if they are made with the same argument. Furthermore, the provided
    module needs to be imported including all submodules specified in
    _REQUIRED_MLIR_MODULES by the caller prior to calling this function. If we
    load_mlir_module is called the first time (i.e. when _mlir_module is still
    None), the given argument is written into _mlir_module (given that the
    required submodules are loaded). If the function is called again, the given
    mlir_module argument needs to be the same each time.
    """
    global _mlir_module
    if _mlir_module and _mlir_module != mlir_module:
        raise RuntimeError('Different modules already loaded previously.')
    for submodule in _REQUIRED_MLIR_MODULES:
        if not hasattr(mlir_module, submodule):
            raise RuntimeError(
                'Provided module "{}" does not have the submodule "{}" loaded.'
                .format(mlir_module.__name__, submodule))
    _mlir_module = mlir_module


def ensure_mlir_module_loaded() -> None:
    """
    Loads the default `mlir` module if no module has been loaded previously by
    `load_mlir_module`.
    """
    if _mlir_module:
        return
    import mlir
    import mlir.ir
    load_mlir_module(mlir)
