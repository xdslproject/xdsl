import argparse
from pathlib import Path

from xdsl.parser import Parser as IRParser
from xdsl.printer import Printer

from .compiler import context, transform
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
    ],
    default="toy-infer-shapes",
    help="Action to perform on source file (default: toy-infer-shapes)",
)
parser.add_argument("--ir", dest="ir", action="store_true")
parser.add_argument("--print-op-generic", dest="print_generic", action="store_true")


def main(path: Path, emit: str, ir: bool, print_generic: bool):
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

    transform(ctx, module_op, target=emit)

    if ir:
        printer = Printer(print_generic_format=print_generic)
        printer.print(module_op)
        return

    interpreter = Interpreter(module_op)
    interpreter.register_implementations(ToyFunctions())
    interpreter.call_op("main", ())

    print(f"Unknown option {emit}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.source, args.emit, args.ir, args.print_generic)
