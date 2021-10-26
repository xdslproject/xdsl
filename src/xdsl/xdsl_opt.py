#!/usr/bin/env python3

import argparse
import sys
from xdsl.ir import *
from xdsl.parser import *
from xdsl.printer import *
from xdsl.dialects.std import *
from xdsl.dialects.scf import *
from xdsl.dialects.affine import *
from xdsl.dialects.builtin import *

arg_parser = argparse.ArgumentParser(
    description='MLIR modular optimizer driver')
arg_parser.add_argument("-f",
                        type=str,
                        required=False,
                        help="path to input file")


def __main__(input_str: str):
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    affine = Affine(ctx)
    scf = Scf(ctx)

    parser = Parser(ctx, input_str)
    module = parser.parse_op()
    module.verify()

    printer = Printer()
    printer.print_op(module)


def main():
    args = arg_parser.parse_args()
    if not args.f:
        input_str = sys.stdin.read()
    else:
        f = open(args.f, mode='r')
        input_str = f.read()

    __main__(input_str)


if __name__ == "__main__":
    main()
