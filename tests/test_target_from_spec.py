from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import IO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.target import Target
from xdsl.xdsl_opt_main import xDSLOptMain


def test_new_style_target_with_args():
    """New-style Target subclasses receive parsed arguments via from_spec."""

    @dataclass(frozen=True)
    class EchoTarget(Target):
        name = "echo"
        prefix: str = ""

        def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
            print(f"{self.prefix}ok", file=output)

    class TestMain(xDSLOptMain):
        def register_all_targets(self):
            self.available_targets["echo"] = lambda: EchoTarget

        def get_input_stream(self) -> tuple[IO[str], str]:
            return (StringIO("builtin.module {}"), "mlir")

    opt = TestMain(args=["-t", "echo"])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    assert "ok" in f.getvalue()

    opt = TestMain(args=["-t", 'echo{prefix="hello "}'])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    assert "hello ok" in f.getvalue()


def test_new_style_target_no_args():
    """New-style Target with no fields works when called as -t name."""

    @dataclass(frozen=True)
    class NoopTarget(Target):
        name = "noop"

        def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
            print("noop", file=output)

    class TestMain(xDSLOptMain):
        def register_all_targets(self):
            self.available_targets["noop"] = lambda: NoopTarget

        def get_input_stream(self) -> tuple[IO[str], str]:
            return (StringIO("builtin.module {}"), "mlir")

    opt = TestMain(args=["-t", "noop"])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    assert "noop" in f.getvalue()
