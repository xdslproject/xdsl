from contextlib import redirect_stdout
from io import StringIO

import pytest
from xdsl.dialects import builtin

from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.xdsl_opt_main import xDSLOptMain


def test_opt():
    opt = xDSLOptMain(args=[])
    assert list(opt.available_frontends.keys()) == ["mlir"]
    assert list(opt.available_targets.keys()) == ["mlir", "riscv-asm"]
    assert list(opt.available_passes.keys()) == [
        "lower-mpi",
        "convert-stencil-to-ll-mlir",
        "convert-stencil-to-gpu",
        "stencil-shape-inference",
        "stencil-to-local-2d-horizontal",
        "frontend-desymrefy",
        "dce",
        "riscv-allocate-registers",
    ]


def test_empty_program():
    filename = "tests/xdsl_opt/empty_program.mlir"
    opt = xDSLOptMain(args=[filename])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()

    with open(filename, "r") as file:
        expected = file.read()
        assert f.getvalue().strip() == expected.strip()


@pytest.mark.parametrize(
    "args, expected_error",
    [
        (["tests/xdsl_opt/not_module.mlir"], "Expected ModuleOp at top level!"),
        (["tests/xdsl_opt/not_module.mlir"], "Expected ModuleOp at top level!"),
        (["tests/xdsl_opt/empty_program.wrong"], "Unrecognized file extension 'wrong'"),
    ],
)
def test_error_on_run(args: list[str], expected_error: str):
    opt = xDSLOptMain(args=args)

    with pytest.raises(Exception) as e:
        opt.run()

    assert expected_error in e.value.args[0]


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

    with open(filename_in, "r") as file:
        inp = file.read()
    with open(filename_out, "r") as file:
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
                    if isinstance(op, builtin.ModuleOp):
                        if op.ops.first is not None:
                            op.ops.first.detach()

            self.register_pass(RemoveConstantPass)

    opt = xDSLOptMainPass(args=[filename_in, "-p", "remove-constant"])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()
    with open(filename_out, "r") as file:
        expected = file.read()

    assert f.getvalue().strip() == expected.strip()


def test_split_input():
    filename_in = "tests/xdsl_opt/split_input_file.mlir"
    filename_out = "tests/xdsl_opt/split_input_file.out"
    flag = "-split-input-file"

    opt = xDSLOptMain(args=[filename_in, flag, "-o", filename_out])
    opt.run()
    with open(filename_in, "r") as file:
        inp = file.read()
    with open(filename_out, "r") as file:
        expected = file.read()

    assert inp.strip() == expected.strip()
