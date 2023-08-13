import ctypes
import subprocess
from pathlib import Path
from typing import Callable, TypeVar

from xdsl.dialects import func
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.ir import Attribute
from xdsl.traits import SymbolTable


def _filename(text: str) -> str:
    module_str = str(text)
    return hex(abs(hash(module_str)))[2:]


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


def _compile_module(module_str: str, directory: Path, filename: str):
    # Lower to LLVM dialect

    mlir_opt_cmd = subprocess.run(
        ["mlir-opt", "--convert-func-to-llvm"],
        input=module_str,
        stdout=subprocess.PIPE,
        text=True,
    )

    mlir_opt_cmd.check_returncode()

    with open(directory / f"{filename}.mlir", "w") as f:
        f.write(mlir_opt_cmd.stdout)

    # Translate MLIR IR to LLVM IR

    mlir_translate_cmd = subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir"],
        input=mlir_opt_cmd.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )

    mlir_translate_cmd.check_returncode()

    with open(directory / f"{filename}.ll", "w") as f:
        f.write(mlir_translate_cmd.stdout)

    # Compile using clang
    # -x ir (LLVM IR as input)
    # -w (ignore all warnings)
    # - (not sure what that does)
    # -shared (we don't need an entry point, this lets us just call the functions)

    mlir_translate_cmd = subprocess.run(
        [
            "clang",
            "-x",
            "ir",
            "-w",
            "-",
            "-shared",
            "-o",
            (directory / f"{filename}.so").as_posix(),
        ],
        input=mlir_translate_cmd.stdout,
        text=True,
    )

    mlir_translate_cmd.check_returncode()


T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


def jit_module(
    module: ModuleOp,
    name: str,
    *,
    c_types: tuple[tuple[type[T0], type[T1]], type[T2]],
) -> Callable[[T0, T1], T2]:
    # check that the types match
    op = SymbolTable.lookup_symbol(module, name)
    if op is None:
        raise ValueError(f"No op with name {name} found in module")

    if not isinstance(op, func.FuncOp):
        raise ValueError(f"Unexpected op type {op.name}, expected func.func")

    module_str = str(module)
    filename = _filename(module_str)

    file_path = Path() / f"{filename}.so"

    if not file_path.exists():
        _compile_module(module_str, Path(), filename)

    libc = ctypes.CDLL(file_path.absolute().as_posix())

    cfunc = libc[name]
    cfunc.argtypes = (ctypes.c_int64, ctypes.c_int64)

    return cfunc
