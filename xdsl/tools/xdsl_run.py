#!/usr/bin/env python3

import argparse
import sys
from collections.abc import Sequence

from xdsl.interpreter import Interpreter
from xdsl.interpreters import (
    register_implementations,
)
from xdsl.ir import MLContext
from xdsl.tools.command_line_tool import CommandLineTool


class xDSLRunMain(CommandLineTool):
    interpreter: Interpreter

    def __init__(
        self,
        description: str = "xDSL modular runner",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}

        self.ctx = MLContext()
        self.register_all_dialects()
        self.register_all_frontends()
        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        arg_parser.add_argument(
            "--wgpu",
            default=False,
            action="store_true",
            help="Enable the WGPU JIT-compilation interpreter.",
        )
        arg_parser.add_argument(
            "--verbose",
            default=False,
            action="store_true",
            help="Print resulting Python values.",
        )
        arg_parser.add_argument(
            "--symbol",
            default="main",
            type=str,
            help="Name of function to call.",
        )
        return super().register_all_arguments(arg_parser)

    def register_implementations(self, interpreter: Interpreter):
        register_implementations(interpreter, self.ctx, self.args.wgpu)

    def run(self):
        input, file_extension = self.get_input_stream()
        try:
            module = self.parse_chunk(input, file_extension)
            if module is not None:
                module.verify()
                interpreter = Interpreter(module)
                self.register_implementations(interpreter)
                symbol = self.args.symbol
                assert isinstance(symbol, str)
                result = interpreter.call_op(symbol, ())
                if self.args.verbose:
                    print(f"result: {result}")
        finally:
            if input is not sys.stdin:
                input.close()


def main():
    return xDSLRunMain().run()


if __name__ == "__main__":
    main()
