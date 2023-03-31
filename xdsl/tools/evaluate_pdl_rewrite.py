import argparse
from enum import Enum, auto
import subprocess

from io import StringIO
from xdsl.dialects.builtin import ModuleOp

from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.passes.pdl_analysis import pdl_analysis_pass
from xdsl.printer import Printer


class AnalysisStatus(Enum):
    SAFE = auto()
    UNSAFE = auto()
    ERROR = auto()


def static_analysis_result(prog: ModuleOp) -> AnalysisStatus:
    s = StringIO()
    printer = Printer(stream=s)
    printer.print_op(prog)
    res = subprocess.run(["xdsl-opt", "-p", "pdl-analysis"],
                         input=s.getvalue(),
                         text=True,
                         capture_output=True)
    if res.returncode != 0:
        print("Error in PDL static analysis: ")
        print(res.stderr)
        return AnalysisStatus.ERROR
    if "UserWarning" in res.stdout:
        return AnalysisStatus.UNSAFE
    return AnalysisStatus.SAFE


def fuzzing_result(prog: ModuleOp, mlir_executable: str) -> AnalysisStatus:
    s = StringIO()
    printer = Printer(stream=s)
    printer.print_op(prog)
    res = subprocess.run([
        "python", "xdsl/tools/pdl_match_fuzzer.py", "--mlir-executable",
        mlir_executable
    ],
                         text=True)
    res.stderr = res.stderr or ""
    if res.returncode == 0:
        assert 'MLIR failed' not in res.stderr
        return AnalysisStatus.SAFE
    if 'MLIR failed' in res.stderr:
        return AnalysisStatus.UNSAFE
    print("Error while fuzzing PDL matchers: ")
    print(res.stderr)
    return AnalysisStatus.ERROR


class EvaluatePDLRewrite(xDSLOptMain):

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)
        arg_parser.add_argument("--mlir-executable", type=str, required=True)

    def run(self):
        module = self.parse_input()
        static_analysis_res = static_analysis_result(module)
        if static_analysis_res == AnalysisStatus.UNSAFE:
            print("PDL static analysis marked the rewrite as unsafe")
        elif static_analysis_res == AnalysisStatus.SAFE:
            print("PDL static analysis marked the rewrite as safe")
        else:
            print("PDL static analysis encountered an error")

        fuzzing_res = fuzzing_result(module, self.args.mlir_executable)
        if static_analysis_res == AnalysisStatus.UNSAFE:
            print("fuzzing marked the rewrite as unsafe")
        elif static_analysis_res == AnalysisStatus.SAFE:
            print("fuzzing marked the rewrite as safe")
        else:
            print("fuzzing encountered an error")

        if fuzzing_res == AnalysisStatus.ERROR or static_analysis_res == AnalysisStatus.ERROR:
            exit(1)

        exit(fuzzing_res != static_analysis_res)


if __name__ == "__main__":
    EvaluatePDLRewrite().run()