import ctypes
import subprocess
from pathlib import Path
from typing import Callable, TypeVar

from xdsl.dialects import func
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.ir import Attribute
from xdsl.traits import SymbolTable

T = TypeVar("T")
U = TypeVar("U")


def jit_types_for_xdsl_type(attr: Attribute) -> tuple[type, type]:
    match attr:
        case IntegerType():
            match attr.width.data:
                case 16:
                    ctypes_type = ctypes.c_int16
                case 32:
                    ctypes_type = ctypes.c_int32
                case 64:
                    ctypes_type = ctypes.c_int64
                case _:
                    raise ValueError(
                        f"Unhandled integer width {attr.width.data} for jit conversion"
                    )

            return int, ctypes_type
        case _:
            raise ValueError(f"Unknown attr type {attr} for jit conversion")


def _compile_module(module: ModuleOp, path: Path):
    # print module

    module_str = str(module)

    # Lower to LLVM dialect

    mlir_opt_cmd = subprocess.run(
        ["mlir-opt", "--convert-func-to-llvm"],
        input=module_str,
        stdout=subprocess.PIPE,
        text=True,
    )

    mlir_opt_cmd.check_returncode()

    # Translate MLIR IR to LLVM IR

    mlir_translate_cmd = subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir"],
        input=mlir_opt_cmd.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )

    mlir_translate_cmd.check_returncode()

    # Compile using clang
    # -x ir (LLVM IR as input)
    # -w (ignore all warnings)
    # - (not sure what that does)
    # -shared (we don't need an entry point, this lets us just call the functions)

    mlir_translate_cmd = subprocess.run(
        ["clang", "-x", "ir", "-w", "-", "-shared", "-o", path.as_posix()],
        input=mlir_translate_cmd.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )

    mlir_translate_cmd.check_returncode()


def jit_module(
    module: ModuleOp,
    name: str,
    *,
    c_types: tuple[tuple[type[int], type[int]], type[int]],
    folder: Path | None = None,
) -> Callable[[int, int], int]:
    """ """
    if folder is None:
        folder = Path()

    # check that the types match
    op = SymbolTable.lookup_symbol(module, name)
    if op is None:
        raise ValueError(f"No op with name {name} found in module")

    if not isinstance(op, func.FuncOp):
        raise ValueError(f"Unexpected op type {op.name}, expected func.func")

    file_path = folder / f"{name}.so"

    if not file_path.exists():
        _compile_module(module, file_path)

    libc = ctypes.CDLL(file_path.absolute().as_posix())

    cfunc = libc[name]
    cfunc.argtypes = (ctypes.c_int64, ctypes.c_int64)

    return cfunc
