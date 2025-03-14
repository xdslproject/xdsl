#!/usr/bin/env python3

import argparse
import sys
from collections.abc import Sequence

from xdsl.context import Context
from xdsl.interpreter import Interpreter
from xdsl.interpreters import (
    register_implementations,
)
from xdsl.parser import Parser
from xdsl.tools.command_line_tool import CommandLineTool
from xdsl.traits import CallableOpInterface


class xDSLRunMain(CommandLineTool):
    interpreter: Interpreter

    def __init__(
        self,
        description: str = "xDSL modular runner",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}

        self.ctx = Context()
        self.register_all_dialects()
        self.register_all_frontends()
        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
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
        arg_parser.add_argument(
            "--index-bitwidth",
            choices=(32, 64),
            # Otherwise default is None, overriding interpreter default mechanism
            default=Interpreter.DEFAULT_BITWIDTH,
            type=int,
            nargs="?",
            help="Bitwidth of the index type representation.",
        )
        arg_parser.add_argument(
            "--args",
            default="",
            type=str,
            help="Arguments to pass to entry function. Comma-separated list of xDSL "
            "Attributes, that will be parsed and converted by the interpreter.",
        )
        return super().register_all_arguments(arg_parser)

    def register_implementations(self, interpreter: Interpreter):
        register_implementations(interpreter, self.ctx)

    def run(self):
        input, file_extension = self.get_input_stream()
        try:
            module = self.parse_chunk(input, file_extension)
            if module is not None:
                module.verify()
                interpreter = Interpreter(
                    module, index_bitwidth=self.args.index_bitwidth
                )
                self.register_implementations(interpreter)
                symbol = self.args.symbol
                assert isinstance(symbol, str)
                parser = Parser(self.ctx, self.args.args, "args")
                runner_args = parser.parse_optional_undelimited_comma_separated_list(
                    parser.parse_optional_attribute, parser.parse_attribute
                )
                if runner_args is None:
                    runner_args = ()
                op = interpreter.get_op_for_symbol(symbol)
                trait = op.get_trait(CallableOpInterface)
                assert trait is not None

                args = tuple(
                    interpreter.value_for_attribute(attr, attr_type)
                    for attr, attr_type in zip(
                        runner_args, trait.get_argument_types(op)
                    )
                )
                result = interpreter.call_op(op, args)
                if self.args.verbose:
                    if result:
                        if len(result) == 1:
                            print(f"result: {result[0]}")
                        else:
                            print("result: (")
                            print(",\n".join(f"    {res}" for res in result))
                            print(")")
                    else:
                        print("result: ()")
        finally:
            if input is not sys.stdin:
                input.close()


def main():
    return xDSLRunMain().run()


if __name__ == "__main__":
    main()
