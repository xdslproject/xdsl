import pytest
from xdsl.xdsl_opt_main import xDSLOptMain
from contextlib import redirect_stdout
from io import StringIO
import os
from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.rewriter import Rewriter


def test_opt():
    opt = xDSLOptMain(args=[])
    assert list(opt.available_frontends.keys()) == ['xdsl', 'mlir']
    assert list(opt.available_targets.keys()) == ['xdsl', 'irdl', 'mlir']
    assert list(opt.available_passes.keys()) == []


def test_empty_program():
    filename = 'tests/xdsl_opt/empty_program.xdsl'
    opt = xDSLOptMain(args=[filename])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()

    expected = open(filename, 'r').read()
    assert f.getvalue().strip() == expected.strip()


@pytest.mark.parametrize("args, expected_error",
                         [(['tests/xdsl_opt/not_module.xdsl'],
                           "Expected module or program as toplevel operation"),
                          (['tests/xdsl_opt/not_module.mlir'],
                           "Expected module or program as toplevel operation"),
                          (['tests/xdsl_opt/empty_program.wrong'
                            ], "Unrecognized file extension 'wrong'")])
def test_error_on_run(args, expected_error):
    opt = xDSLOptMain(args=args)

    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[0] == expected_error


@pytest.mark.parametrize(
    "args, expected_error",
    [(['tests/xdsl_opt/empty_program.xdsl', '-p', 'wrong'
       ], "Unrecognized pass: wrong")])
def test_error_on_construction(args, expected_error):
    with pytest.raises(Exception) as e:
        opt = xDSLOptMain(args=args)

    assert e.value.args[0] == expected_error


def test_wrong_target():
    filename = 'tests/xdsl_opt/empty_program.xdsl'
    opt = xDSLOptMain(args=[filename])
    opt.args.target = "wrong"

    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[0] == "Unknown target wrong"


def test_print_to_file():
    filename_in = 'tests/xdsl_opt/empty_program.xdsl'
    filename_out = 'tests/xdsl_opt/empty_program.out'
    opt = xDSLOptMain(args=[filename_in, '-o', filename_out])

    inp = open(filename_in, 'r').read()
    opt.run()
    expected = open(filename_out, 'r').read()

    assert inp.strip() == expected.strip()


def test_operation_deletion():
    filename_in = 'tests/xdsl_opt/constant_program.xdsl'
    filename_out = 'tests/xdsl_opt/empty_program.xdsl'

    class xDSLOptMainPass(xDSLOptMain):

        def register_all_passes(self):

            def remove_constant(ctx: MLContext, module: ModuleOp):
                module.ops[0].detach()

            self.available_passes['remove-constant'] = remove_constant

    opt = xDSLOptMainPass(args=[filename_in, '-p', 'remove-constant'])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()
    expected = open(filename_out, 'r').read()

    assert f.getvalue().strip() == expected.strip()
