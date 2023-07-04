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

from .frontend.ir_gen import IRGen
from .frontend.parser import Parser as ToyParser
from .compiler import context, emulate_riscv, transform

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
        "toy",
        "toy-opt",
        "toy-inline",
        "toy-infer-shapes",
        "affine",
        "scf",
        "cf",
        "riscv",
        "riscv-regalloc",
        "riscv-assembly",
        "riscemu",
    ],
    default="riscemu",
    help="Action to perform on source file (default: riscemu)",
)
parser.add_argument("--ir", dest="ir", action="store_true")
parser.add_argument("--print-op-generic", dest="print_generic", action="store_true")
parser.add_argument("--accelerate", dest="accelerate", action="store_true")


def main(path: Path, emit: str, ir: bool, accelerate: bool, print_generic: bool):
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

    riscemu = emit == "riscemu"
    print_assembly = emit == "riscv-assembly"

    if riscemu or print_assembly:
        emit = "riscv-regalloc"

    transform(ctx, module_op, target=emit, accelerate=accelerate)

    if ir:
        printer = Printer(print_generic_format=print_generic)
        printer.print(module_op)
        return

    if riscemu or print_assembly:
        code = riscv.riscv_code(module_op)

        if print_assembly:
            print(code)
            return

        if riscemu:
            emulate_riscv(code)
            return

    if emit == "riscv-regalloc":
        print("Interpretation of register allocated code currently unsupported")
        # The reason is that we lower functions before register allocation, and lose
        # the mechanism of function calls in the interpreter.
        return

    interpreter = Interpreter(module_op)
    if emit in ("toy", "toy-opt", "toy-inline", "toy-infer-shapes"):
        interpreter.register_implementations(ToyFunctions())
    if emit in ("affine"):
        interpreter.register_implementations(AffineFunctions())
    if accelerate and emit in ("affine", "scf", "cf"):
        interpreter.register_implementations(ToyAcceleratorFunctions())
    if emit in ("affine", "scf", "cf"):
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(MemrefFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())
    if emit == "scf":
        interpreter.register_implementations(ScfFunctions())
    if emit == "cf":
        interpreter.register_implementations(CfFunctions())

    if emit in ("scf, cf"):

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

    if emit in ("riscv",):
        interpreter.register_implementations(ToyAcceleratorInstructionFunctions())
        interpreter.register_implementations(RiscvCfFunctions())
        interpreter.register_implementations(RiscvFuncFunctions())
        interpreter.register_implementations(BuiltinFunctions())

    interpreter.call_op("main", ())

    print(f"Unknown option {emit}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.source, args.emit, args.ir, args.accelerate, args.print_generic)
