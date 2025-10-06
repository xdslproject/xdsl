import ctypes
import subprocess

import llvmlite
import llvmlite.binding as llvm
import llvmlite.ir as ir

from xdsl.dialects import func
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    IntegerType,
    ModuleOp,
    ParametrizedAttribute,
)
from xdsl.ir import Block


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


def xdsl_to_llvmlite_types(xdsl_type):
    if not isinstance(xdsl_type, IntegerType):
        raise Exception("Currently only integer conversion is supported")

    return ir.IntType(xdsl_type.bitwidth)


def translate_to_llvm_lite_block(
    xdsl_block: Block, llvm_lite_block: llvmlite.ir.values.Block, ssa_translation: dict
):
    builder = ir.IRBuilder(llvm_lite_block)

    for op in xdsl_block.ops:
        inputs = [ssa_translation[operand] for operand in op.operands]

        if isinstance(op, func.ReturnOp):
            builder.ret(*inputs)


def translate_to_llvmlite_module(
    xdsl_module: ModuleOp, use_parser: bool = True
) -> llvmlite.binding.module.ModuleRef:
    if not use_parser:
        # Create an LLVM module
        llvm_lite_module = ir.Module(name="xdsl_module")

        for op in xdsl_module.ops:
            if isinstance(op, func.FuncOp):
                # Translate function signature
                xdsl_outputs = op.properties["function_type"].outputs.data
                xdsl_inputs = op.properties["function_type"].inputs.data
                llvm_lite_outpus = [
                    xdsl_to_llvmlite_types(xdsl_type) for xdsl_type in xdsl_outputs
                ]
                llvm_lite_inputs = [
                    xdsl_to_llvmlite_types(xdsl_type) for xdsl_type in xdsl_inputs
                ]

                # Create function type
                func_type = ir.FunctionType(llvm_lite_outpus[0], llvm_lite_inputs)

                # Create function operation
                func_name = op.properties["sym_name"].data
                llvm_lite_func = ir.Function(
                    llvm_lite_module, func_type, name=func_name
                )

                # Translate function body
                xdsl_block = op.regions[0].blocks[0]

                # Set up SSA value translation dictionary
                ssa_translation = {
                    xdsl_block.args[i]: llvm_lite_func.args[i]
                    for i in range(len(xdsl_block.args))
                }

                llvm_lite_block = llvm_lite_func.append_basic_block(name="entry")
                translate_to_llvm_lite_block(
                    xdsl_block, llvm_lite_block, ssa_translation
                )

            else:
                raise Exception(
                    "Tried to convert operation type {type(op} in top-level module to LLVM"
                )

        return llvm_lite_module

    else:
        mlir_text = str(xdsl_module)
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

        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()

        return mod


def llvm_jit(module: ModuleOp, kernel_func_name: str) -> callable:
    for op in module.walk():
        if isinstance(op, func.FuncOp):
            if op.properties["sym_name"].data == kernel_func_name:
                break
    else:
        raise Exception("Couldn't find kernel in provided module")

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

    llvm_lite_module = translate_to_llvmlite_module(module)

    engine.add_module(llvm_lite_module)
    engine.finalize_object()
    engine.run_static_constructors()
    func_ptr = engine.get_function_address(kernel_func_name)
    cfunc_type = ctypes.CFUNCTYPE(*(out_types + in_types))
    cfunc = cfunc_type(func_ptr)

    # --- Closure retains refs to engine/mod ---
    def wrapper(*args):
        return cfunc(*args)

    wrapper._engine = engine  # Prevent GC
    wrapper._mod = llvm_lite_module  # Prevent GC

    return wrapper
