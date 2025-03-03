import dis
import marshal


def view_pyc_file(path):
    """Read and display a content of the Python`s bytecode in a pyc-file."""

    with open(path, "rb") as f:
        f.seek(16)
        code_object = marshal.load(f)

    dis.disassemble(code_object)


view_pyc_file(
    "/Users/edjg/Desktop/thesis/code/xdsl-bench/xdsl/xdsl/irdl/__pycache__/operations.cpython-312.pyc"
)
