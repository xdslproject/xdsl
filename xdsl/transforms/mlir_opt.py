import shutil
import subprocess
from dataclasses import dataclass, field
from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import DiagnosticException, ParseError


@dataclass(frozen=True)
class MLIROptPass(ModulePass):
    """
    A pass for calling the `mlir-opt` tool with specified parameters. Will fail if
    `mlir-opt` is not available.
    """

    name = "mlir-opt"

    executable: str = field(default="mlir-opt")
    generic: bool = field(default=True)
    arguments: tuple[str, ...] = field(default=())

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        if not shutil.which(self.executable):
            raise ValueError(f"{self.executable} is not available")

        stream = StringIO()
        printer = Printer(print_generic_format=self.generic, stream=stream)
        printer.print_op(op)

        my_string = stream.getvalue()

        completed_process = subprocess.run(
            [self.executable, *self.arguments],
            input=my_string,
            capture_output=True,
            text=True,
        )

        try:
            completed_process.check_returncode()
        except subprocess.CalledProcessError as e:
            raise DiagnosticException("Error executing mlir-opt pass") from e

        # Get the stdout output
        stdout_output = completed_process.stdout
        parser = Parser(ctx, stdout_output)

        try:
            new_module = parser.parse_module()
        except ParseError as e:
            raise DiagnosticException("Error parsing mlir-opt pass output") from e

        op.detach_region(op.body)
        op.add_region(Rewriter().move_region_contents_to_new_regions(new_module.body))
        op.attributes = new_module.attributes
