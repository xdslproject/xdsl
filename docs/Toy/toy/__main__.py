import argparse
from pathlib import Path
from typing import Any

from xdsl.dialects import memref, riscv
from xdsl.dialects.builtin import Float64Type, IndexType, IntegerType
from xdsl.interpreter import InterpreterFunctions, impl_cast, register_impls
from xdsl.interpreters.affine import AffineFunctions
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.memref import MemrefFunctions
from xdsl.interpreters.printf import PrintfFunctions
from xdsl.interpreters.riscv import Buffer
from xdsl.interpreters.riscv_func import RiscvFuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from .compiler import context, transform
from .emulator.toy_accelerator_functions import ToyAcceleratorFunctions
from .emulator.toy_accelerator_instruction_functions import (
    ShapedArrayBuffer,
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
    ],
    default="riscv",
    help="Action to perform on source file (default: riscv)",
)
parser.add_argument("--ir", dest="ir", action="store_true")
parser.add_argument("--print-op-generic", dest="print_generic", action="store_true")
parser.add_argument("--accelerate", dest="accelerate", action="store_true")


@register_impls
class BufferMemrefConversion(InterpreterFunctions):
    @impl_cast(riscv.IntRegisterType, memref.MemRefType)
    def cast_buffer_to_memref(
        self,
        input_type: riscv.IntRegisterType,
        output_type: memref.MemRefType[Float64Type],
        value: Any,
    ) -> Any:
        shape = output_type.get_shape()
        return ShapedArrayBuffer(value.data, list(shape))

    @impl_cast(memref.MemRefType, riscv.IntRegisterType)
    def cast_memref_to_buffer(
        self,
        input_type: memref.MemRefType[Float64Type],
        output_type: riscv.IntRegisterType,
        value: Any,
    ) -> Any:
        return Buffer(value.data)

    @impl_cast(IndexType, riscv.IntRegisterType)
    def cast_index_to_int_reg(
        self,
        input_type: IndexType,
        output_type: riscv.IntRegisterType,
        value: Any,
    ) -> Any:
        return value

    # Hack for partial lowering and lack of support for float registers
    @impl_cast(Float64Type, riscv.IntRegisterType)
    def cast_float_to_int_reg(
        self,
        input_type: Float64Type,
        output_type: riscv.IntRegisterType,
        value: Any,
    ) -> Any:
        return value

    # Hack for partial lowering and lack of support for float registers
    @impl_cast(riscv.IntRegisterType, Float64Type)
    def cast_int_reg_to_float(
        self,
        input_type: riscv.IntRegisterType,
        output_type: Float64Type,
        value: Any,
    ) -> Any:
        return float(value)

    @impl_cast(riscv.IntRegisterType, IntegerType)
    def cast_reg_to_int(
        self,
        input_type: riscv.IntRegisterType,
        output_type: IntegerType,
        value: Any,
    ) -> Any:
        return value

    @impl_cast(IntegerType, riscv.IntRegisterType)
    def cast_int_to_reg(
        self,
        input_type: IntegerType,
        output_type: riscv.IntRegisterType,
        value: Any,
    ) -> Any:
        return value

    @impl_cast(IndexType, riscv.IntRegisterType)
    def cast_index_to_reg(
        self,
        input_type: IndexType,
        output_type: riscv.IntRegisterType,
        value: Any,
    ) -> Any:
        return value

    @impl_cast(riscv.IntRegisterType, IndexType)
    def cast_reg_to_index(
        self,
        input_type: riscv.IntRegisterType,
        output_type: IndexType,
        value: Any,
    ) -> Any:
        return value

    # Below casts temporary workaround for lack of float support in the pipeline

    @impl_cast(IntegerType, Float64Type)
    def cast_int_to_float(
        self,
        input_type: IntegerType,
        output_type: Float64Type,
        value: Any,
    ) -> Any:
        return float(value)

    @impl_cast(Float64Type, IntegerType)
    def cast_float_to_int(
        self,
        input_type: Float64Type,
        output_type: IntegerType,
        value: Any,
    ) -> Any:
        return int(value)


def main(path: Path, emit: str, ir: bool, accelerate: bool, print_generic: bool):
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

    transform(ctx, module_op, target=emit, accelerate=accelerate)

    if ir:
        printer = Printer(print_generic_format=print_generic)
        printer.print(module_op)
        return

    interpreter = Interpreter(module_op)
    if emit in ("toy", "toy-opt", "toy-inline", "toy-infer-shapes"):
        interpreter.register_implementations(ToyFunctions())
    if emit in ("affine"):
        interpreter.register_implementations(AffineFunctions())
    if accelerate and emit in ("affine", "scf"):
        interpreter.register_implementations(ToyAcceleratorFunctions())
    if emit in ("affine", "scf"):
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(MemrefFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())
    if emit == "scf":
        interpreter.register_implementations(ScfFunctions())
        interpreter.register_implementations(BuiltinFunctions())

    if emit in ("riscv",):
        interpreter.register_implementations(
            ToyAcceleratorInstructionFunctions(module_op)
        )
        interpreter.register_implementations(BufferMemrefConversion())
        interpreter.register_implementations(RiscvFuncFunctions())
        interpreter.register_implementations(BuiltinFunctions())
        # TODO: remove as we add lowerings to riscv
        interpreter.register_implementations(ScfFunctions())
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(PrintfFunctions())
        interpreter.register_implementations(FuncFunctions())

    interpreter.call_op("main", ())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.source, args.emit, args.ir, args.accelerate, args.print_generic)
