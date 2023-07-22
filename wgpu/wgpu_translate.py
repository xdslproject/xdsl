#!/usr/bin/env python3

from io import StringIO
from xdsl.dialects.builtin import ModuleOp
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.interpreters.experimental.wgsl_printer import WGSLPrinter


class WGPUTranslateMain(xDSLOptMain):
    def output_resulting_program(self, prog: ModuleOp) -> str:
        """Get the resulting program."""
        output = StringIO()
        wgpu_translator = WGSLPrinter()
        wgpu_translator.print(prog.ops.first, output)
        return output.getvalue()


def main():
    WGPUTranslateMain().run()


if __name__ == "__main__":
    main()
