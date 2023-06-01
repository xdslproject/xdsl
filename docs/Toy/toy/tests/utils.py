from io import StringIO
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    riscv.print_assembly(module, stream)
    return stream.getvalue()
