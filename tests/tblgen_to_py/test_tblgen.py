from io import StringIO

from xdsl.tools.tblgen_to_py import tblgen_to_py


def test_run_tblgen_to_py():
    with StringIO() as output:
        tblgen_to_py("tests/tblgen_to_py/test.json", output)
        out_str = output.getvalue()

    with open("tests/tblgen_to_py/test.py") as f:
        expected = f.read()

        assert len(out_str.strip()) == len(expected.strip())
        assert out_str.strip() == expected.strip()
