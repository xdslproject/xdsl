from contextlib import redirect_stdout
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import IO

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.mlir import MLIRFrontend
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError
from xdsl.utils.frontend import Frontend
from xdsl.xdsl_opt_main import xDSLOptMain


def test_new_style_frontend_with_args():
    """New-style Frontend subclasses receive parsed arguments via from_spec."""

    @dataclass(frozen=True)
    class EchoFrontend(Frontend):
        name = "echo"
        prefix: str = ""

        def parse(self, ctx: Context, input: IO[bytes], input_name: str) -> ModuleOp:
            # Return a simple module with the prefix determining the module body
            if self.prefix:
                return Parser(
                    ctx,
                    f"builtin.module {{ func.func @{self.prefix}() {{}} }}",
                    input_name,
                ).parse_module()
            else:
                return Parser(
                    ctx, "builtin.module { func.func @nop() {} }", input_name
                ).parse_module()

    class TestMain(xDSLOptMain):
        def register_all_frontends(self):
            self.available_frontends["echo"] = lambda: EchoFrontend

        def get_input_stream(self) -> tuple[IO[bytes], str]:
            return (BytesIO(b""), "echo")

    opt = TestMain(args=["-f", "echo"])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    output = f.getvalue()
    assert "@nop" in output

    opt = TestMain(args=["-f", 'echo{prefix="hello"}'])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    output = f.getvalue()
    assert "@hello" in output


def test_new_style_frontend_no_args():
    """New-style Frontend with no fields works when called as -f name."""

    @dataclass(frozen=True)
    class SimpleFrontend(Frontend):
        name = "simple"

        def parse(self, ctx: Context, input: IO[bytes], input_name: str) -> ModuleOp:
            return Parser(ctx, "builtin.module {}", input_name).parse_module()

    class TestMain(xDSLOptMain):
        def register_all_frontends(self):
            self.available_frontends["simple"] = lambda: SimpleFrontend

        def get_input_stream(self) -> tuple[IO[bytes], str]:
            return (BytesIO(b""), "simple")

    opt = TestMain(args=["-f", "simple"])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    # Just check it doesn't crash


def test_mlir_frontend_no_implicit_module_flag():
    """--no-implicit-module flag is passed to MLIRFrontend."""

    class TestMain(xDSLOptMain):
        def register_all_frontends(self):
            self.available_frontends["mlir"] = lambda: MLIRFrontend

        def get_input_stream(self) -> tuple[IO[bytes], str]:
            # Provide MLIR without explicit module - only works with implicit module
            return (BytesIO(b"func.func @test() {}"), "mlir")

    # Without --no-implicit-module, should work (implicit module wraps the function)
    opt = TestMain(args=[])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    assert "func.func @test" in f.getvalue()

    # With --no-implicit-module, should fail (no module wrapper)
    opt = TestMain(args=["--no-implicit-module"])
    f = StringIO()
    with redirect_stdout(f):
        with pytest.raises(ParseError):
            opt.run()


def test_mlir_frontend_with_valid_input():
    """MLIRFrontend correctly parses valid input."""

    class TestMain(xDSLOptMain):
        def register_all_frontends(self):
            self.available_frontends["mlir"] = lambda: MLIRFrontend

        def get_input_stream(self) -> tuple[IO[bytes], str]:
            return (BytesIO(b"builtin.module {}"), "mlir")

    opt = TestMain(args=[])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    assert "builtin.module" in f.getvalue()


def test_legacy_frontend_still_works():
    """Legacy frontend function (IO[bytes]) -> ModuleOp still works."""

    class TestMain(xDSLOptMain):
        def register_all_frontends(self):
            # Register a legacy-style frontend (function, not factory)
            def legacy_frontend(io: IO[bytes]) -> ModuleOp:
                # Use self.ctx which is available in the method scope
                return Parser(self.ctx, b"builtin.module {}", "legacy").parse_module()

            self.available_frontends["legacy"] = legacy_frontend

        def get_input_stream(self) -> tuple[IO[bytes], str]:
            return (BytesIO(b""), "legacy")

    opt = TestMain(args=["-f", "legacy"])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    # Just check it doesn't crash and produces output
    assert "builtin.module" in f.getvalue()


def test_legacy_frontend_ignores_args():
    """Legacy frontend silently ignores spec arguments (matches target behavior)."""

    class TestMain(xDSLOptMain):
        def register_all_frontends(self):
            # Register a legacy-style frontend (function, not factory)
            def legacy_frontend(io: IO[bytes]) -> ModuleOp:
                return Parser(self.ctx, b"builtin.module {}", "legacy").parse_module()

            self.available_frontends["legacy"] = legacy_frontend

        def get_input_stream(self) -> tuple[IO[bytes], str]:
            return (BytesIO(b""), "legacy")

    # Should not crash even though we pass args to legacy frontend
    opt = TestMain(args=["-f", "legacy{ignored_arg=true}"])
    f = StringIO()
    with redirect_stdout(f):
        opt.run()
    # Just check it doesn't crash and produces output
    assert "builtin.module" in f.getvalue()
