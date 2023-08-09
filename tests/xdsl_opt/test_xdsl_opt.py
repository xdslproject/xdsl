from contextlib import redirect_stdout
from io import StringIO

import pytest

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.utils.exceptions import DiagnosticException
from xdsl.xdsl_opt_main import get_all_passes, xDSLOptMain


def test_dialects_and_passes():
    assert len(get_all_dialects()) > 0
    assert len(get_all_passes()) > 0


def test_opt():
    opt = xDSLOptMain(args=[])
    assert len(opt.available_frontends.keys()) > 0
    assert len(opt.available_targets.keys()) > 0
    assert len(opt.available_passes.keys()) > 0


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
    "args, expected_error",
    [
        (
            ["--no-implicit-module", "tests/xdsl_opt/not_module.mlir"],
            "builtin.module operation expected",
        ),
        (
            ["--no-implicit-module", "tests/xdsl_opt/incomplete_program.mlir"],
            "Could not parse entire input",
        ),
        (
            ["tests/xdsl_opt/incomplete_program_residual.mlir"],
            "Could not parse entire input",
        ),
        (
            ["tests/xdsl_opt/incomplete_program.mlir"],
            "Could not parse entire input",
        ),
        (["tests/xdsl_opt/empty_program.wrong"], "Unrecognized file extension 'wrong'"),
    ],
)
def test_error_on_run(args: list[str], expected_error: str):
    opt = xDSLOptMain(args=args)

    with pytest.raises(Exception, match=expected_error):
        opt.run()


@pytest.mark.parametrize(
    "args, expected_error",
    [
        (
            ["tests/xdsl_opt/empty_program.mlir", "-p", "wrong"],
            "Unrecognized pass: wrong",
        )
    ],
)
def test_error_on_construction(args: list[str], expected_error: str):
    with pytest.raises(Exception) as e:
        _opt = xDSLOptMain(args=args)

    assert e.value.args[0] == expected_error


def test_wrong_target():
    filename = "tests/xdsl_opt/empty_program.mlir"
    opt = xDSLOptMain(args=[filename])
    opt.args.target = "wrong"

    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[0] == "Unknown target wrong"


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

                def apply(self, ctx: MLContext, op: builtin.ModuleOp):
                    if op.ops.first is not None:
                        op.ops.first.detach()

            self.register_pass(RemoveConstantPass)

    opt = xDSLOptMainPass(args=[filename_in, "-p", "remove-constant"])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()
    with open(filename_out) as file:
        expected = file.read()

    assert f.getvalue().strip() == expected.strip()


def test_print_between_passes():
    filename_in = "tests/xdsl_opt/empty_program.mlir"
    passes = ["stencil-shape-inference", "dce", "frontend-desymrefy"]
    flags = ["--print-between-passes", "-p", ",".join(passes)]

    f = StringIO("")

    opt = xDSLOptMain(args=[*flags, filename_in])

    with redirect_stdout(f):
        opt.run()

    output = f.getvalue()
    assert (
        len([l for l in output.split("\n") if "builtin.module" in l]) == len(passes) + 1
    )


def test_diagnostic_exception():
    filename_in = "tests/xdsl_opt/unverified_program.mlir"

    opt = xDSLOptMain(args=[filename_in])

    with pytest.raises(DiagnosticException):
        opt.run()


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
