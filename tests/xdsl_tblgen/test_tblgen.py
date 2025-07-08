from io import StringIO

from xdsl.tools.xdsl_tblgen import tblgen_to_py


def test_run_tblgen_to_py():
    with StringIO() as output:
        tblgen_to_py("tests/xdsl_tblgen/test.json", output)
        out_str = output.getvalue()

    with open("tests/xdsl_tblgen/test.py") as f:
        expected = f.read()

        assert out_str.strip() == expected.strip()
