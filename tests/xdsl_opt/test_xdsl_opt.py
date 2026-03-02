"""
This test file needs the other files in the same folder to read and reprint them in the
test functions below.
"""

import re
from contextlib import redirect_stdout
from io import StringIO
from typing import IO

import pytest

from xdsl.context import Context
from xdsl.dialects import builtin, get_all_dialects
from xdsl.passes import ModulePass
from xdsl.transforms import get_all_passes
from xdsl.utils.exceptions import DiagnosticException, ParseError
from xdsl.xdsl_opt_main import xDSLOptMain


def test_dialects_and_passes():
    assert get_all_dialects()
    assert get_all_passes()


def test_opt():
    opt = xDSLOptMain(args=[])
    assert opt.available_frontends.keys()
    assert opt.available_targets.keys()
    assert opt.available_passes.keys()


def test_empty_program():
    filename = "tests/xdsl_opt/empty_program.mlir"
    opt = xDSLOptMain(args=[filename])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()

    with open(filename) as file:
        expected = file.read()
        assert f.getvalue().strip() == expected.strip()


@pytest.mark.parametrize(
    "args, error_type, expected_error",
    [
        (
            ["--no-implicit-module", "tests/xdsl_opt/not_module.mlir"],
            ParseError,
            "builtin.module operation expected",
        ),
        (
            ["--no-implicit-module", "tests/xdsl_opt/incomplete_program.mlir"],
            ParseError,
            "Could not parse entire input",
        ),
        (
            ["tests/xdsl_opt/incomplete_program_residual.mlir"],
            ParseError,
            "Could not parse entire input",
        ),
        (
            ["tests/xdsl_opt/incomplete_program.mlir"],
            ParseError,
            "Could not parse entire input",
        ),
        (
            ["tests/xdsl_opt/empty_program.wrong"],
            ValueError,
            "Unrecognized file extension 'wrong'",
        ),
        (
            ["tests/xdsl_opt/unverified_program.mlir"],
            DiagnosticException,
            "operand 'lhs' at position 0 does not verify",
        ),
    ],
)
def test_error_on_run(
    args: list[str], error_type: type[Exception], expected_error: str
):
    opt = xDSLOptMain(args=args)

    with pytest.raises(error_type, match=expected_error):
        opt.run()


@pytest.mark.parametrize(
    "args, error_type, expected_error",
    [
        (
            ["tests/xdsl_opt/empty_program.mlir", "-p", "wrong"],
            ValueError,
            "Unrecognized passes: ['wrong']",
        ),
    ],
)
def test_error_on_construction(
    args: list[str], error_type: type[Exception], expected_error: str
):
    with pytest.raises(error_type, match=re.escape(expected_error)):
        xDSLOptMain(args=args)


@pytest.mark.parametrize(
    "args, expected_error",
    [
        (
            ["tests/xdsl_opt/empty_program.mlir", "-t", "wrong"],
            "invalid choice: 'wrong'",
        ),
    ],
)
def test_error_on_argparse(
    capsys: pytest.CaptureFixture[str], args: list[str], expected_error: str
):
    with pytest.raises(SystemExit):
        xDSLOptMain(args=args)
    out, err = capsys.readouterr()
    assert out == ""
    assert expected_error in err


def test_print_to_file():
    filename_in = "tests/xdsl_opt/empty_program.mlir"
    filename_out = "tests/xdsl_opt/empty_program.out"

    opt = xDSLOptMain(args=[filename_in, "-o", filename_out])
    opt.run()

    with open(filename_in) as file:
        inp = file.read()
    with open(filename_out) as file:
        expected = file.read()

    assert inp.strip() == expected.strip()


def test_operation_deletion():
    filename_in = "tests/xdsl_opt/simple_program.mlir"
    filename_out = "tests/xdsl_opt/empty_program.mlir"

    class xDSLOptMainPass(xDSLOptMain):
        def register_all_passes(self):
            class RemoveConstantPass(ModulePass):
                name = "remove-constant"

                def apply(self, ctx: Context, op: builtin.ModuleOp):
                    if op.ops.first is not None:
                        op.ops.first.detach()

            self.register_pass("remove-constant", lambda: RemoveConstantPass)

    opt = xDSLOptMainPass(args=[filename_in, "-p", "remove-constant"])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()
    with open(filename_out) as file:
        expected = file.read()

    assert f.getvalue().strip() == expected.strip()


def test_print_between_passes():
    filename_in = "tests/xdsl_opt/empty_program.mlir"
    passes = ["shape-inference", "dce", "frontend-desymrefy"]
    flags = ["--print-between-passes", "-p", ",".join(passes)]

    f = StringIO("")

    opt = xDSLOptMain(args=[*flags, filename_in])

    with redirect_stdout(f):
        opt.run()

    output = f.getvalue()
    assert len([l for l in output.split("\n") if "builtin.module" in l]) == len(passes)


def test_verify_diagnostics_output():
    """
    Diagnostic exceptions raised when printing output should be redirected to stdout if
    --verify-diagnostics flag is provided.
    """

    class TestMain(xDSLOptMain):
        def register_all_targets(self):
            def _my_target(prog: builtin.ModuleOp, output: IO[str]):
                raise DiagnosticException("fail")

            self.available_targets["fail"] = _my_target

        def get_input_stream(self) -> tuple[IO[str], str]:
            fake_input = StringIO("builtin.module {}")
            return (fake_input, "mlir")

    opt = TestMain(args=["-t", "fail"])
    f = StringIO("")
    with redirect_stdout(f):
        with pytest.raises(DiagnosticException, match="fail"):
            opt.run()
    assert f.getvalue() == ""

    opt = TestMain(args=["--verify-diagnostics", "-t", "fail"])
    f = StringIO("")
    with redirect_stdout(f):
        opt.run()
    assert f.getvalue() == "fail\n"


def test_split_input():
    filename_in = "tests/xdsl_opt/empty_program.mlir"
    filename_out = "tests/xdsl_opt/split_input_file.out"
    flag = "--split-input-file"

    opt = xDSLOptMain(args=[filename_in, flag, "-o", filename_out])
    opt.run()
    with open(filename_in) as file:
        inp = file.read()
    with open(filename_out) as file:
        expected = file.read()

    assert inp.strip() == expected.strip()
