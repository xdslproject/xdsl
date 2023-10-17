import argparse
from pathlib import Path

from xdsl.dialects.riscv import riscv_code
from xdsl.interpreters.affine import AffineFunctions
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.memref import MemrefFunctions
from xdsl.interpreters.printf import PrintfFunctions
from xdsl.interpreters.riscv_func import RiscvFuncFunctions
from xdsl.interpreters.riscv_scf import RiscvScfFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from .compiler import context, emulate_riscv, transform
from .emulator.toy_accelerator_instruction_functions import (
    ToyAcceleratorInstructionFunctions,
)
from .frontend.ir_gen import IRGen
from .frontend.parser import Parser as ToyParser
from .interpreter import Interpreter, ToyFunctions

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
        "riscv",
        "riscv-opt",
        "riscv-regalloc",
        "riscv-regalloc-opt",
        "riscv-lowered",
        "riscv-asm",
    ],
    default="riscv-asm",
    help="Compilation target (default: riscv-asm)",
)
parser.add_argument("--ir", dest="ir", action="store_true")
parser.add_argument("--print-op-generic", dest="print_generic", action="store_true")


def main(path: Path, emit: str, ir: bool, print_generic: bool):
    ctx = context()

    path = args.source

    with open(path) as f:
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

    asm = emit == "riscv-asm"

    if asm:
        emit = "riscv-lowered"

    transform(ctx, module_op, target=emit)

    if asm:
        code = riscv_code(module_op)

        if ir:
            print(code)
            return

        emulate_riscv(code)
        return

    if ir:
        printer = Printer(print_generic_format=print_generic)
        printer.print(module_op)
        return

    if emit == "riscv-lowered":
        print("Interpretation of lowered riscv code currently unsupported")
        # The reason is that we lower functions before register allocation, and lose
        # the mechanism of function calls in the interpreter.
        return

    interpreter = Interpreter(module_op)
    if emit in ("toy", "toy-opt", "toy-inline", "toy-infer-shapes"):
        interpreter.register_implementations(ToyFunctions())
    if emit in ("affine"):
        interpreter.register_implementations(AffineFunctions())
    if emit in ("affine", "scf"):
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(MemrefFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())
    if emit == "scf":
        interpreter.register_implementations(ScfFunctions())
        interpreter.register_implementations(BuiltinFunctions())

    if emit in ("riscv", "riscv-opt", "riscv-regalloc", "riscv-regalloc-opt"):
        interpreter.register_implementations(ToyAcceleratorInstructionFunctions())
        interpreter.register_implementations(RiscvFuncFunctions())
        interpreter.register_implementations(RiscvScfFunctions())

    interpreter.call_op("main", ())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.source, args.emit, args.ir, args.print_generic)
