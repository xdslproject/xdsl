import argparse

from pathlib import Path
from typing import Any
from xdsl.dialects.builtin import Float64Type

from xdsl.interpreters.affine import AffineFunctions
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.memref import MemrefFunctions
from xdsl.interpreters.printf import PrintfFunctions
from xdsl.interpreters.riscv_cf import RiscvCfFunctions
from xdsl.interpreters.riscv import Buffer
from xdsl.interpreters.riscv_func import RiscvFuncFunctions
from xdsl.interpreters.scf import ScfFunctions

from xdsl.dialects import memref, riscv

from xdsl.printer import Printer
from xdsl.parser import Parser as IRParser
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation

from .frontend.ir_gen import IRGen
from .frontend.parser import Parser as ToyParser
from .compiler import context, emulate_riscv
from .rewrites.optimise_toy import OptimiseToy
from .rewrites.shape_inference import ShapeInferencePass
from .rewrites.inline_toy import InlineToyPass
from .rewrites.lower_toy_affine import LowerToAffinePass
from .rewrites.lower_to_toy_accelerator import (
    LowerToToyAccelerator,
    LowerToyAccelerator,
)
from .rewrites.mlir_opt import MLIROptPass
from .rewrites.setup_riscv_pass import FinalizeRiscvPass, SetupRiscvPass
from .rewrites.lower_printf_riscv import LowerPrintfRiscvPass

from .rewrites.lower_riscv_cf import LowerCfRiscvCfPass

from .rewrites.lower_scf_riscv import LowerScfRiscvPass
from .rewrites.lower_arith_riscv import LowerArithRiscvPass
from .rewrites.lower_memref_riscv import LowerMemrefToRiscv
from .rewrites.lower_func_riscv_func import LowerFuncToRiscvFunc

from .interpreter import Interpreter, ToyFunctions

from .emulator.toy_accelerator_instruction_functions import (
    ShapedArrayBuffer,
    ToyAcceleratorInstructionFunctions,
)
from .emulator.toy_accelerator_functions import ToyAcceleratorFunctions


parser = argparse.ArgumentParser(description="Process Toy file")
parser.add_argument("source", type=Path, help="toy source file")
parser.add_argument(
    "--emit",
    dest="emit",
    choices=[
        "ast",
        "ir-toy",
        "ir-toy-opt",
        "ir-toy-inline",
        "ir-toy-infer-shapes",
        "interpret-toy",
        "ir-affine",
        "interpret-affine",
        "ir-scf",
        "interpret-scf",
        "ir-cf",
        "interpret-cf",
        "ir-riscv",
        "interpret-riscv",
        "riscv-assembly",
        "riscemu",
    ],
    default="interpret",
    help="Action to perform on source file (default: interpret)",
)
parser.add_argument("--print-op-generic", dest="print_generic", action="store_true")
parser.add_argument("--accelerate", dest="accelerate", action="store_true")


def main(path: Path, emit: str, accelerate: bool, print_generic: bool):
    ctx = context()

    path = args.source

    with open(path, "r") as f:
        match path.suffix:
            case ".toy":
                parser = ToyParser(path, f.read())
                ast = parser.parseModule()
                if emit == "ast":
                    print(ast.dump())
                    return

                ir_gen = IRGen()
                module_op = ir_gen.ir_gen_module(ast)
            case ".mlir":
                parser = IRParser(ctx, f.read(), name=f"{path}")
                module_op = parser.parse_module()
            case _:
                print(f"Unknown file format {path}")
                return

    printer = Printer(print_generic_format=print_generic)

    if emit == "ir-toy":
        printer.print(module_op)
        return

    OptimiseToy().apply(ctx, module_op)

    if emit == "ir-toy-opt":
        printer.print(module_op)
        return

    InlineToyPass().apply(ctx, module_op)

    if emit == "ir-toy-inline":
        printer.print(module_op)
        return

    ShapeInferencePass().apply(ctx, module_op)

    if emit == "ir-toy-infer-shapes":
        printer.print(module_op)
        return

    if emit == "interpret-toy":
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(ToyFunctions())
        interpreter.call_op("main", ())
        return

    LowerToAffinePass().apply(ctx, module_op)
    module_op.verify()

    if accelerate:
        LowerToToyAccelerator().apply(ctx, module_op)
        module_op.verify()

    if emit == "ir-affine":
        printer.print(module_op)
        return

    if emit == "interpret-affine":
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(AffineFunctions())
        interpreter.register_implementations(ToyAcceleratorFunctions())
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(MemrefFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())
        interpreter.call_op("main", ())
        return

    MLIROptPass(
        [
            "--allow-unregistered-dialect",
            "--canonicalize",
            "--cse",
            "--lower-affine",
            "--mlir-print-op-generic",
        ]
    ).apply(ctx, module_op)

    if emit == "ir-scf":
        printer.print(module_op)
        return

    if emit == "interpret-scf":
        interpreter = Interpreter(module_op)

        def memfer_from_buffer(
            o: riscv.RegisterType, r: memref.MemRefType[Float64Type], value: Any
        ) -> Any:
            shape = r.get_shape()
            return ShapedArrayBuffer(value.data, list(shape))

        def buffer_from_memref(
            o: memref.MemRefType[Float64Type], r: riscv.RegisterType, value: Any
        ) -> Any:
            return Buffer(value.data)

        builtin_functions = BuiltinFunctions()

        builtin_functions.register_cast_impl(
            riscv.RegisterType, memref.MemRefType, memfer_from_buffer
        )
        builtin_functions.register_cast_impl(
            memref.MemRefType, riscv.RegisterType, buffer_from_memref
        )
        interpreter.register_implementations(builtin_functions)
        interpreter.register_implementations(ScfFunctions())
        interpreter.register_implementations(ToyAcceleratorFunctions())
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(MemrefFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())
        interpreter.register_implementations(ToyAcceleratorInstructionFunctions())
        interpreter.call_op("main", ())
        return

    MLIROptPass(
        [
            "--allow-unregistered-dialect",
            "--canonicalize",
            "--cse",
            "--convert-scf-to-cf",
            "--mlir-print-op-generic",
        ]
    ).apply(ctx, module_op)

    if emit == "ir-cf":
        printer.print(module_op)
        return

    if emit == "interpret-cf":
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(CfFunctions())
        interpreter.register_implementations(ToyAcceleratorFunctions())
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(MemrefFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())
        interpreter.call_op("main", ())
        return

    SetupRiscvPass().apply(ctx, module_op)
    LowerFuncToRiscvFunc().apply(ctx, module_op)
    LowerCfRiscvCfPass().apply(ctx, module_op)
    LowerToyAccelerator().apply(ctx, module_op)
    LowerScfRiscvPass().apply(ctx, module_op)
    DeadCodeElimination().apply(ctx, module_op)
    LowerArithRiscvPass().apply(ctx, module_op)
    LowerPrintfRiscvPass().apply(ctx, module_op)
    LowerMemrefToRiscv().apply(ctx, module_op)
    FinalizeRiscvPass().apply(ctx, module_op)

    DeadCodeElimination().apply(ctx, module_op)

    module_op.verify()

    if emit == "ir-riscv":
        printer.print(module_op)
        return

    if emit == "interpret-riscv":
        interpreter = Interpreter(module_op)

        interpreter.register_implementations(ToyAcceleratorInstructionFunctions())
        interpreter.register_implementations(RiscvCfFunctions())
        interpreter.register_implementations(RiscvFuncFunctions())
        interpreter.register_implementations(BuiltinFunctions())

        interpreter.call_op("main", ())
        return

    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)
    RISCVRegisterAllocation().apply(ctx, module_op)

    module_op.verify()

    code = riscv.riscv_code(module_op)

    if emit == "riscv-assembly":
        print(code)
        return

    if emit == "riscemu":
        emulate_riscv(code)
        return

    print(f"Unknown option {emit}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.source, args.emit, args.accelerate, args.print_generic)
