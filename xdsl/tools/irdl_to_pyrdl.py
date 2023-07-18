"""Script to convert an IRDL program to an xDSL dialect in Python."""

import argparse
import sys

from xdsl.dialects.irdl import DialectOp
from xdsl.dialects.irdl.irdl_to_pyrdl import convert_dialect
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.tools.command_line_tool import get_all_dialects


def main():
    # Parse CLI arguments
    arg_parser = argparse.ArgumentParser(
        description="Convert an IRDL program to a Python definition of a xDSL dialect."
    )
    arg_parser.add_argument(
        "-o", "--output-file", type=str, required=False, help="path to output file"
    )
    arg_parser.add_argument("input_file", type=str, help="path to input file")
    args = arg_parser.parse_args()

    ctx = MLContext()
    dialects = get_all_dialects()
    for dialect in dialects:
        ctx.register_dialect(dialect)

    # Parse the input file
    f = open(args.input_file)
    parser = Parser(ctx, f.read(), name=args.input_file)
    module = parser.parse_module()

    # Prepare the output file
    if args.output_file is None:
        file = sys.stdout
    else:
        file = open(args.output_file, "w")

    # Output the Python code
    print("from xdsl.irdl import *", file=file)
    print("from xdsl.ir import *\n\n", file=file)
    for op in module.walk():
        if isinstance(op, DialectOp):
            print(convert_dialect(op), file=file)


if __name__ == "__main__":
    main()
