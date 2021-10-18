import pytest

docutils = pytest.importorskip("mlir")
from xdsl.mlir_converter import *
from xdsl.dialects.memref import *

test_prog = """
module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %1 : !memref<[1 : !i64], !i32> = memref.alloca() ["alignment" = 0 : !i64, "operand_segment_sizes" = !vector<[0 : !i64, 0 : !i64]>]

    std.return()
  }
}
"""


def test_conversion():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    affine = Affine(ctx)
    scf = Scf(ctx)
    memref = MemRef(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()
    module.verify()
    printer = Printer()
    printer.print_op(module)

    converter = MLIRConverter(ctx)
    with mlir.ir.Context() as mlir_ctx:
        mlir_ctx.allow_unregistered_dialects = True
        with mlir.ir.Location.unknown(mlir_ctx):
            mlir_affine_func = converter.convert_op(module)
            print(mlir_affine_func)


if __name__ == "__main__":
    test_conversion()
