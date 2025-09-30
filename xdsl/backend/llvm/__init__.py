import ctypes
import subprocess

import llvmlite.binding as llvm

from xdsl.dialects import func
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    IntegerType,
    ModuleOp,
    ParametrizedAttribute,
)


def xdsl_to_ctypes(xdsl_type: ParametrizedAttribute):
    if xdsl_type == IntegerType(64):
        return ctypes.c_longlong
    if xdsl_type == IntegerType(32):
        return ctypes.c_int32
    if xdsl_type == IntegerType(16):
        return ctypes.c_int16
    if xdsl_type == IntegerType(8):
        return ctypes.c_int8
    if xdsl_type == IntegerType(1):
        return ctypes.c_bool
    if xdsl_type == Float32Type():
        return ctypes.c_float
    if xdsl_type == Float64Type():
        return ctypes.c_double

    raise TypeError(f"Unsupported or unknown xDSL type: {xdsl_type}")


def translate_to_llvm(
    module: ModuleOp,
) -> (
    type(ctypes.c_longlong)
    | type(ctypes.c_int32)
    | type(ctypes.c_int16)
    | type(ctypes.c_int8)
    | type(ctypes.c_bool)
    | type(ctypes.c_float)
    | type(ctypes.c_double)
):
    mlir_text = str(module)
    mlir_opt_passes = [
        "mlir-opt",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--expand-strided-metadata",
        "--normalize-memrefs",
        "--memref-expand",
        "--fold-memref-alias-ops",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
        "-",
    ]
    mlir_opt = subprocess.Popen(
        mlir_opt_passes, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    mlir_translate = subprocess.Popen(
        ["mlir-translate", "--mlir-to-llvmir", "-"],
        stdin=mlir_opt.stdout,
        stdout=subprocess.PIPE,
    )
    mlir_opt.stdin.write(mlir_text.encode())
    mlir_opt.stdin.close()
    llvm_ir_bytes, _ = mlir_translate.communicate()
    llvm_ir = llvm_ir_bytes.decode()

    return llvm_ir


def llvm_jit(module: ModuleOp, kernel_func_name: str) -> callable:
    for op in module.walk():
        if isinstance(op, func.FuncOp):
            if op.properties["sym_name"].data == kernel_func_name:
                break
    else:
        raise Exception("Couldn't find kernel in provided module")

    llvm_ir = translate_to_llvm(module)

    in_types = [
        xdsl_to_ctypes(op_type) for op_type in op.properties["function_type"].inputs
    ]
    out_types = [
        xdsl_to_ctypes(ret_type) for ret_type in op.properties["function_type"].outputs
    ]

    # --- Keep the engine and mod alive! ---
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    func_ptr = engine.get_function_address(kernel_func_name)
    cfunc_type = ctypes.CFUNCTYPE(*(out_types + in_types))
    cfunc = cfunc_type(func_ptr)

    # --- Closure retains refs to engine/mod ---
    def wrapper(*args):
        return cfunc(*args)

    wrapper._engine = engine  # Prevent GC
    wrapper._mod = mod  # Prevent GC

    return wrapper
