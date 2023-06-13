import os
import sys
from xdsl.dialects.builtin import IndexType, StringAttr
from xdsl.dialects.func import FuncOp
from xdsl.interpreter import Interpreter
from xdsl.runner.arith import ArithFunctions
from xdsl.runner.builtin import BuiltinFunctions
from xdsl.runner.func import FuncFunctions
from xdsl.utils.hints import isa
from xdsl.xdsl_opt_main import xDSLOptMain


class xDSLRunMain(xDSLOptMain):
    def run(self):
        if self.args.input_file is None:
            input = sys.stdin
            file_extension = "mlir"
        else:
            input = open(self.args.input_file)
            _, file_extension = os.path.splitext(self.args.input_file)
            file_extension = file_extension.replace(".", "")
        try:
            module = self.parse_chunk(input, file_extension)
            if module is not None:
                if self.apply_passes(module):
                    interpreter = Interpreter(module)
                    interpreter.register_implementations(BuiltinFunctions())
                    interpreter.register_implementations(FuncFunctions())
                    interpreter.register_implementations(ArithFunctions())
                    interpreter.run(module)
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
                            for ins in f.body.ops:
                                interpreter.run(ins)
                            return interpreter.get_values(last.operands)[0]
        finally:
            if input is not sys.stdout:
                input.close()


def main():
    return xDSLRunMain().run()
