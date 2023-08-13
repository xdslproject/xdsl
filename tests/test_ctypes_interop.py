from pathlib import Path

# from tempfile import TemporaryDirectory
# import ctypes
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import ModuleOp, i64


def get_module() -> ModuleOp:
    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        f = func.FuncOp("hello", ((i64, i64), (i64,)))
        with ImplicitBuilder(f.body) as (lhs, rhs):
            res = arith.Addi(lhs, rhs).result
            func.Return(res)

    return module


def write_module(module: ModuleOp, base: Path, name: str):
    with open(base / f"{name}.mlir", "x") as f:
        f.write(f"{module}")


def compile_module(base: Path, name: str):
    ...


# def main(base: Path):
#     module = get_module()
#     module_name = "my_add"
#     write_module(module, base, module_name)
#     compile_module(base, module_name)
#     run_function(base, module_name)


# def test_run_addition():
#     with TemporaryDirectory("xdsl_tmp") as tmp_dir:
#         path = Path(tmp_dir)
#         main(base)

#         print(tmp_dir)
#         print(path)

#     assert 1 == 1
