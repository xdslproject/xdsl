from xdsl.xdsl_opt_main import xDSLOptMain
from contextlib import redirect_stdout
from io import StringIO
import pytest


def test_opt():
    opt = xDSLOptMain()
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


def test_not_module_xdsl():
    filename = 'tests/xdsl_opt/not_module.xdsl'
    opt = xDSLOptMain(args=[filename])

    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[
        0] == "Expected module or program as toplevel operation"


def test_not_module_mlir():
    filename = 'tests/xdsl_opt/not_module.mlir'
    opt = xDSLOptMain(args=[filename])

    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[
        0] == "Expected module or program as toplevel operation"


def test_print_to_file():
    filename_in = 'tests/xdsl_opt/empty_program.xdsl'
    filename_out = 'tests/xdsl_opt/empty_program_out.xdsl'
    opt = xDSLOptMain(args=[filename_in, '-o', filename_out])

    inp = open(filename_in, 'r').read()
    opt.run()
    expected = open(filename_out, 'r').read()

    assert inp.strip() == expected.strip()

    open(filename_out, 'w').write("")


def test_wrong_target():
    filename_in = 'tests/xdsl_opt/empty_program.xdsl'
    opt = xDSLOptMain(args=[filename_in])
    opt.args.target = "wrong"

    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[0] == "Unknown target wrong"


def test_wrong_pass():
    filename_in = 'tests/xdsl_opt/empty_program.xdsl'
    with pytest.raises(Exception) as e:
        opt = xDSLOptMain(args=[filename_in, '-p', 'wrong'])

    assert e.value.args[0] == "Unrecognized pass: wrong"


def test_wrong_file_extension():
    filename_in = 'tests/xdsl_opt/empty_program.wrong'
    opt = xDSLOptMain(args=[filename_in])

    f = StringIO("")
    with pytest.raises(Exception) as e:
        opt.run()

    assert e.value.args[0] == "Unrecognized file extension 'wrong'"
