import argparse
import dataclasses
import inspect
import os
import sys
from collections.abc import Callable
from typing import IO, cast

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.utils.arg_spec import parse_spec
from xdsl.utils.frontend import Frontend


class CommandLineTool:
    ctx: Context
    args: argparse.Namespace
    """
    The argument parsers namespace which holds the parsed commandline
    attributes.
    """

    available_frontends: dict[
        str, Callable[[IO[bytes]], ModuleOp] | Callable[[], type[Frontend]]
    ]
    """
    A mapping from file extension to a frontend that can handle this
    file type.
    """

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        arg_parser.add_argument(
            "input_file", type=str, nargs="?", help="path to input file"
        )

        frontends = [name for name in self.available_frontends]
        arg_parser.add_argument(
            "-f",
            "--frontend",
            type=str,
            required=False,
            choices=frontends,
            help="Frontend to be used for the input. If not set, "
            "the xdsl frontend or the one for the file extension "
            "is used.",
        )
        arg_parser.add_argument("--disable-verify", default=False, action="store_true")

        arg_parser.add_argument(
            "--allow-unregistered-dialect",
            default=False,
            action="store_true",
            help="Allow the parsing of unregistered dialects.",
        )

        arg_parser.add_argument(
            "--no-implicit-module",
            default=False,
            action="store_true",
            help="Disable implicit addition of a top-level module op during parsing.",
        )

    def get_input_stream(self) -> tuple[IO[bytes], str]:
        """
        Get the binary input stream to parse from, along with the file extension.
        """
        if self.args.input_file is None:
            f = sys.stdin.buffer
            file_extension = "mlir"
        else:
            f = open(self.args.input_file, "rb")
            _, file_extension = os.path.splitext(self.args.input_file)
            file_extension = file_extension.replace(".", "")
        return f, file_extension

    def get_input_name(self):
        return self.args.input_file or "stdin"

    def register_all_dialects(self):
        """
        Register all dialects that can be used.

        Add other/additional dialects by overloading this function.
        """
        from xdsl.universe import Universe

        multiverse = Universe.get_multiverse()
        for dialect_name, dialect_factory in multiverse.all_dialects.items():
            self.ctx.register_dialect(dialect_name, dialect_factory)

    def register_all_frontends(self):
        """
        Register all frontends that can be used.

        Add other/additional frontends by overloading this function.
        """

        def parse_mlir(io: IO[bytes]):
            return Parser(
                self.ctx,
                io.read(),
                self.get_input_name(),
            ).parse_module(not self.args.no_implicit_module)

        self.available_frontends["mlir"] = parse_mlir

    def parse_chunk(
        self, chunk: IO[bytes], file_extension: str, start_offset: int = 0
    ) -> ModuleOp | None:
        """
        Parse the input file by invoking the frontend specified.
        Uses self.args.frontend if provided, otherwise falls back to file_extension.
        """
        # Determine frontend key (from -f flag or file extension)
        frontend_key = self.args.frontend or file_extension

        # Parse spec (handles both "mlir" and "mlir{arg=val}")
        frontend_spec = parse_spec(frontend_key)
        frontend_entry = self.available_frontends[frontend_spec.name]
        sig = inspect.signature(frontend_entry)

        if not sig.parameters:
            # New-style: factory -> Frontend class
            factory = cast(Callable[[], type[Frontend]], frontend_entry)
            frontend = factory().from_spec(frontend_spec)

            # Handle --no-implicit-module for MLIR frontend (backward compat, overrides spec)
            if frontend_spec.name == "mlir":
                frontend = dataclasses.replace(
                    frontend,
                    no_implicit_module=self.args.no_implicit_module,
                )

            return frontend.parse(self.ctx, chunk)
        else:
            # Legacy: function (IO[bytes]) -> ModuleOp (spec args silently ignored)
            legacy = cast(Callable[[IO[bytes]], ModuleOp], frontend_entry)
            return legacy(chunk)
