"""Script to convert an IRDL program to an xDSL dialect in Python."""

import argparse

from xdsl.dialects.irdl import DialectOp
from xdsl.dialects.irdl.irdl_to_pyrdl import convert_dialect
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.xdsl_opt_main import get_all_dialects


def main():
    arg_parser = argparse.ArgumentParser(
        description="Convert an IRDL program to a Python definition of a xDSL dialect."
    )
    arg_parser.add_argument("input_file", type=str, help="path to input file")
    args = arg_parser.parse_args()

    ctx = MLContext()
    dialects = get_all_dialects()
    for dialect in dialects:
        ctx.register_dialect(dialect)

    # Parse the input file
    args.input_file
    f = open(args.input_file)
    parser = Parser(ctx, f.read(), name=args.input_file)
    module = parser.parse_module()

    print("from xdsl.irdl import *")
    print("from xdsl.ir import *\n\n")
    for op in module.walk():
        if isinstance(op, DialectOp):
            print(convert_dialect(op))


if __name__ == "__main__":
    main()
