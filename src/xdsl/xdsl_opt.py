#!/usr/bin/env python3

import argparse
import sys
import os
from xdsl.ir import *
from xdsl.parser import *
from xdsl.printer import *
from xdsl.dialects.func import *
from xdsl.dialects.scf import *
from xdsl.dialects.arith import *
from xdsl.dialects.affine import *
from xdsl.dialects.memref import *
from xdsl.dialects.builtin import *
from xdsl.dialects.cf import *


class xDSLOptMain:
    ctx: MLContext
    args: argparse.Namespace

    def __init__(self, args: argparse.Namespace):
        self.ctx = MLContext()
        self.args = args

    def register_all_dialects(self):
        """Register all dialects that can be used."""
        builtin = Builtin(self.ctx)
        func = Func(self.ctx)
        arith = Arith(self.ctx)
        memref = MemRef(self.ctx)
        affine = Affine(self.ctx)
        scf = Scf(self.ctx)
        cf = Cf(self.ctx)

    def parse_frontend(self) -> ModuleOp:
        """Parse the input file."""
        if self.args.input_file is None:
            f = sys.stdin
            file_extension = '.xdsl'
        else:
            f = open(self.args.input_file, mode='r')
            _, file_extension = os.path.splitext(self.args.input_file)

        if file_extension == '.xdsl':
            input_str = f.read()
            parser = Parser(self.ctx, input_str)
            module = parser.parse_op()
            if not self.args.disable_verify:
                module.verify()
            if not (isinstance(module, ModuleOp)):
                raise Exception(
                    "Expected module or program as toplevel operation")
            return module

        raise Exception(f"Unrecognized file extension '{file_extension}'")

    def output_resulting_program(self, prog: ModuleOp) -> str:
        """Get the resulting program."""
        output = StringIO()
        if self.args.target == 'xdsl':
            printer = Printer(stream=output)
            printer.print_op(prog)
            return output.getvalue()
        if self.args.target == 'mlir':
            try:
                from xdsl.mlir_converter import MLIRConverter
            except ImportError as ex:
                raise Exception(
                    "Can only emit mlir if the mlir bindings are present"
                ) from ex
            converter = MLIRConverter(self.ctx)
            mlir_module = converter.convert_module(prog)
            print(mlir_module, file=output)
            return output.getvalue()
        raise Exception(f"Unknown target {self.args.target}")

    def print_to_output_stream(self, contents: str):
        """Print the contents in the expected stream."""
        if self.args.output_file is None:
            print(contents)
        else:
            output_stream = open(self.args.output_file, 'w')
            output_stream.write(contents)


arg_parser = argparse.ArgumentParser(
    description='MLIR modular optimizer driver')
arg_parser.add_argument("input_file",
                        type=str,
                        nargs="?",
                        help="path to input file")

arg_parser.add_argument("-t",
                        "--target",
                        type=str,
                        required=False,
                        choices=["xdsl", "mlir"],
                        help="target",
                        default="xdsl")

arg_parser.add_argument("--disable-verify", default=False, action='store_true')
arg_parser.add_argument("-o",
                        "--output-file",
                        type=str,
                        required=False,
                        help="path to output file")


def __main__(args: argparse.Namespace):
    xdsl_main = xDSLOptMain(args)

    xdsl_main.register_all_dialects()
    module = xdsl_main.parse_frontend()

    contents = xdsl_main.output_resulting_program(module)
    xdsl_main.print_to_output_stream(contents)


def main():
    args = arg_parser.parse_args()

    __main__(args)


if __name__ == "__main__":
    main()
