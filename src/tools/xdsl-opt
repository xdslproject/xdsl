#!/usr/bin/env python3

import argparse
from xdsl.xdsl_opt_main import xDSLOptMain


class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        super().register_all_dialects()

    def register_all_passes(self):
        super().register_all_passes()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)


def __main__():
    xdsl_main = OptMain()

    module = xdsl_main.parse_input()

    contents = xdsl_main.output_resulting_program(module)
    xdsl_main.print_to_output_stream(contents)


if __name__ == "__main__":
    __main__()
