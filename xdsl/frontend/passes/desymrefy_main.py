import io
import os
import sys

from argparse import ArgumentParser, Namespace
from xdsl.dialects.arith import Arith

from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.frontend.passes.desymref import DesymrefyPass
from xdsl.frontend.symref import Symref
from xdsl.ir import MLContext
from xdsl.parser import XDSLParser
from xdsl.printer import Printer


class DesymrefyMain:
    ctx: MLContext
    args: Namespace

    def __init__(self):
        self.ctx = MLContext()
        self.ctx.register_dialect(Builtin)
        self.ctx.register_dialect(Func)
        self.ctx.register_dialect(Arith)
        self.ctx.register_dialect(Symref)

        arg_parser = ArgumentParser(
            description="Driver to test desymrefication pass.")
        arg_parser.add_argument("input_file",
                                type=str,
                                nargs="?",
                                help="Path to input file.")
        arg_parser.add_argument("-o",
                                "--output-file",
                                type=str,
                                required=False,
                                help="Path to output file.")
        self.args = arg_parser.parse_args()

    def run(self):
        # Process input.
        if self.args.input_file is None:
            f = sys.stdin
            file_extension = "xdsl"
        else:
            f = open(self.args.input_file)
            _, file_extension = os.path.splitext(self.args.input_file)
            file_extension = file_extension.replace(".", "")

        # Parse xDSL module.
        module = XDSLParser(self.ctx, f.read(), self.args.input_file
                            or "stdin").parse_module()
        assert isinstance(module, ModuleOp)

        # Run desymrefication and verify the module is correct.
        DesymrefyPass.run(module)
        module.verify()

        # Process the output.
        output = io.StringIO()
        Printer(stream=output).print_op(module)
        if self.args.output_file is None:
            print(output.getvalue())
        else:
            output_stream = open(self.args.output_file, 'w')
            output_stream.write(output.getvalue())
