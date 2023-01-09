from xdsl.xdsl_opt_main import xDSLOptMain
from contextlib import redirect_stdout
from io import StringIO


def test_opt():
    filename = 'tests/xdsl_opt/empty_program.xdsl'
    opt = xDSLOptMain(args=[filename])
    assert list(opt.available_frontends.keys()) == ['xdsl', 'mlir']
    assert list(opt.available_targets.keys()) == ['xdsl', 'irdl', 'mlir']
    assert list(opt.available_passes.keys()) == []

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()

    expected = open(filename, 'r').read()
    assert f.getvalue().strip() == expected.strip()
