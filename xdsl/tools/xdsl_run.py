#!/usr/bin/env python3

import argparse
import sys
from xdsl.dialects.builtin import IndexType, StringAttr
from xdsl.dialects.func import FuncOp
from xdsl.interpreter import Interpreter
from xdsl.runner.arith import ArithFunctions
from xdsl.runner.builtin import BuiltinFunctions
from xdsl.runner.func import FuncFunctions
from xdsl.runner.print import PrintFunctions
from xdsl.runner.scf import ScfFunctions
from xdsl.utils.hints import isa
from xdsl.xdsl_opt_main import xDSLOptMain


class xDSLRunMain(xDSLTool):
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

    def register_implementations(self, interpreter: Interpreter):
        interpreter.register_implementations(func.FuncFunctions())
        interpreter.register_implementations(arith.ArithFunctions())
        interpreter.register_implementations(printf.PrintfFunctions())
        interpreter.register_implementations(scf.ScfFunctions())

    def run(self):
        input, file_extension = self.get_input_stream()
        try:
            module = self.parse_chunk(input, file_extension)
            if module is not None:
                if self.apply_passes(module):
                    interpreter = Interpreter(module)
                    interpreter.register_implementations(BuiltinFunctions())
                    interpreter.register_implementations(FuncFunctions())
                    interpreter.register_implementations(ArithFunctions())
                    interpreter.register_implementations(PrintFunctions())
                    interpreter.register_implementations(ScfFunctions())
                    for f in module.walk():
                        if isinstance(f, FuncOp) and f.sym_name == StringAttr("main"):
                            assert (
                                last := f.body.ops.last
                            ) is not None, f"Expected an internal main symbol"
                            assert (
                                len(last.operands) == 1
                            ), f"Expected main symbol's terminator to have 1 operand"
                            assert isa(
                                last.operands[0].typ, IndexType
                            ), f"Expected main symbol's terminator to have an index operand"
                            ret = interpreter.run_ssacfg_region(f.body, ())
                            return ret[0]
        finally:
            if input is not sys.stdin:
                input.close()


def main():
    return xDSLRunMain().run()


if __name__ == "__main__":
    main()
