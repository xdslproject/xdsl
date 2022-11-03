from math import ceil
from xdsl.dialects.elevate.dialect import ReturnOp
from xdsl.ir import MLContext, OpResult
from xdsl.dialects.arith import Addi, Arith, Constant, AndI
from xdsl.dialects.func import *
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.onnx.dialect as onnx
import xdsl.dialects.stencil.stencil as stencil
from xdsl.dialects.builtin import Builtin, IntAttr, IntegerAttr, IntegerType, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer
import argparse
from sys import stdin
import statistics


def main():
    arg_parser = argparse.ArgumentParser(prog='', description='')
    arg_parser.add_argument("input_file",
                            type=argparse.FileType('r'),
                            nargs="?",
                            default=stdin)
    arg_parser.add_argument("--print", default=False, action='store_true')

    args = arg_parser.parse_args()

    program = args.input_file.read()
    program = program.replace('\"module\"', '\"builtin.module\"')
    program = program.replace('\"func\"', '\"func.func\"')
    program = program.replace('?x?x?xf64', '[70 : i64, 70 : i64, 70 : i64]')
    program = program.replace('?', '70')
    program = program.replace('std.constant', 'arith.constant')
    program = program.replace('std.addf', 'arith.addf')
    program = program.replace('std.divf', 'arith.divf')
    program = program.replace('std.mulf', 'arith.mulf')
    program = program.replace('std.subf', 'arith.subf')
    program = program.replace('std.cmpf', 'arith.cmpf')
    program = program.replace('std.negf', 'arith.negf')
    program = program.replace('std.select', 'arith.select')
    program = program.replace('std.return', 'func.return')
    program = program.replace('type = ', 'function_type = ')
    program = program.replace('\"module.module_terminator',
                              '//\"module.module_terminator')
    program = program.replace('\"module_terminator', '//\"module_terminator')
    program = program.replace('x', ',')
    print(program)


if __name__ == "__main__":
    main()
