from conftest import assert_print_op

from xdsl.dialects.builtin import Builtin
from xdsl.dialects.gpu import GPU
from xdsl.ir import MLContext
from xdsl.parser import Parser


def parse_and_compare(prog: str, expected_prog: str):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(GPU)

    parser = Parser(ctx, prog)

    # import pdb

    # pdb.set_trace()
    module = parser.parse_module()

    print(module)

    assert_print_op(module, expected_prog, None)


def test_gpu1():
    prog = """
"builtin.module"() ({
    "gpu.module"() ({
      ^0:
    }) {"sym_name" = "gpu"} : () -> ()
}) {"gpu.container_module"} : () -> ()
"""

    expected = """
"builtin.module"() ({
    "gpu.module"() ({
      "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {"gpu.container_module"} : () -> ()
"""

    parse_and_compare(prog, expected)
