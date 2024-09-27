import subprocess


def test_run_tblgen_to_py():
    output = subprocess.run(
        [
            "xdsl-tblgen",
            "-i",
            "tests/tblgen_to_py/test.json",
        ],
        capture_output=True,
        text=True,
    )

    out_str = output.stdout

    with open("tests/tblgen_to_py/test.py") as f:
        expected = f.read()

        assert len(out_str.strip()) == len(expected.strip())
        assert out_str.strip() == expected.strip()
