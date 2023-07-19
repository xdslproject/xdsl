import shutil
import subprocess
from dataclasses import dataclass, field
from io import StringIO

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.parser.core import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.printer import Printer
from xdsl.utils.exceptions import DiagnosticException


@dataclass
class MLIROptPass(ModulePass):
    """
    A pass for calling the `mlir-opt` tool with specified parameters. Will fail if
    `mlir-opt` is not available.
    """

    name = "mlir-opt"

    arguments: list[str] = field(default_factory=list)

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        if not shutil.which("mlir-opt"):
            raise ValueError("mlir-opt is not available")

        stream = StringIO()
        printer = Printer(print_generic_format=True, stream=stream)
        printer.print(op)

        my_string = stream.getvalue()

        completed_process = subprocess.run(
            ["mlir-opt", *self.arguments],
            input=my_string,
            stdout=subprocess.PIPE,
            text=True,
        )

        try:
            completed_process.check_returncode()

            # Get the stdout output
            stdout_output = completed_process.stdout

            parser = Parser(ctx, stdout_output)

            new_module = parser.parse_module()

            rewriter = PatternRewriter(op)
            op.detach_region(op.body)
            op.add_region(rewriter.move_region_contents_to_new_regions(new_module.body))
        except Exception as e:
            raise DiagnosticException("Error executing mlir-opt pass") from e
