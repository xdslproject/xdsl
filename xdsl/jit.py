import ctypes
import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

from xdsl.dialects import func, memref
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    IntegerType,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Attribute
from xdsl.traits import SymbolTable


def _filename(text: str) -> str:
    m = hashlib.sha256()
    m.update(text.encode())
    return m.hexdigest()


ScalarType = (
    ctypes.c_int16 | ctypes.c_int32 | ctypes.c_int64 | ctypes.c_double | ctypes.c_float
)


def c_type_for_xdsl_scalar_type(attr: Attribute) -> type[ScalarType]:
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

            return ctypes_type
        case Float64Type():
            return ctypes.c_double
        case Float32Type():
            return ctypes.c_float
        case _:
            raise ValueError(f"Unknown attr type {attr} for jit conversion")


if TYPE_CHECKING:
    Pointer: TypeAlias = ctypes._Pointer[ScalarType]  # pyright: ignore[reportPrivateUsage]


def c_type_for_xdsl_type(attr: Attribute) -> "type[ScalarType] | type[Pointer]":
    if isinstance(attr, MemRefType):
        attr = cast(memref.MemRefType[Attribute], attr)
        # TODO: check that shape is fully known
        c_type = c_type_for_xdsl_scalar_type(attr.element_type)
        res = ctypes.POINTER(c_type)
        return res
    else:
        return c_type_for_xdsl_scalar_type(attr)


def _compile_module(module_str: str, directory: Path, filename: str):
    input_name = (directory / f"{filename}_in.mlir").as_posix()
    output_name = (directory / f"{filename}_out.mlir").as_posix()

    with open(input_name, "w") as f:
        f.write(module_str)

    # Lower to LLVM dialect

    cmd = (
        f"mlir-opt {input_name} "
        # + "--test-lower-to-llvm"
        + "--convert-scf-to-cf "
        + "--canonicalize "
        + "--cse "
        + "--convert-arith-to-llvm "
        + "--finalize-memref-to-llvm='use-generic-functions' "
        + "--convert-func-to-llvm='use-bare-ptr-memref-call-conv' "
        + "--convert-index-to-llvm "
        + "--convert-cf-to-llvm "
        + "--convert-arith-to-llvm "
        + "--reconcile-unrealized-casts "
        + f" > {output_name}"
    )

    # print(cmd)

    os.system(cmd)

    # Translate MLIR IR to LLVM IR

    mlir_translate_cmd = subprocess.run(
        [
            "mlir-translate",
            "--mlir-to-llvmir",
            output_name,
        ],
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

    clang_cmd = subprocess.run(
        [
            "clang",
            "-O3",
            "-x",
            "ir",
            "-w",
            "-",
            "-shared",
            "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
            "-o",
            (directory / f"{filename}.so").as_posix(),
        ],
        input=mlir_translate_cmd.stdout,
        text=True,
    )

    clang_cmd.check_returncode()


T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


def jit_module(
    module: ModuleOp,
    name: str,
    *,
    types: tuple[tuple[type[T0], type[T1]], type[T2]],
) -> Callable[[T0, T1], T2]:
    # check that the types match
    op = SymbolTable.lookup_symbol(module, name)
    if op is None:
        raise ValueError(f"No op with name {name} found in module")

    if not isinstance(op, func.FuncOp):
        raise ValueError(f"Unexpected op type {op.name}, expected func.func")

    input_type_attrs = op.function_type.inputs.data
    result_type_attrs = op.function_type.outputs.data

    if len(result_type_attrs) != 1:
        raise ValueError(
            f"Function type can only have one result value, got {result_type_attrs}"
        )

    argtypes = tuple(c_type_for_xdsl_type(attr) for attr in input_type_attrs)
    restype = c_type_for_xdsl_type(result_type_attrs[0])

    module_str = str(module)
    filename = _filename(module_str)

    file_path = Path() / f"{filename}.so"

    if not file_path.exists():
        try:
            _compile_module(module_str, Path(), filename)
        except Exception as error:
            print(error)
            raise

    libc = ctypes.CDLL(file_path.absolute().as_posix())

    cfunc = libc[name]
    cfunc.argtypes = argtypes
    cfunc.restype = restype

    return cfunc
