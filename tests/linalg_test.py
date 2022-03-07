from xdsl.ir import *
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.scf import Scf
from xdsl.dialects.std import Std
from xdsl.dialects.memref import MemRef
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.linalg import Linalg
from xdsl.parser import Parser
from xdsl.printer import Printer


def parse_and_verify(test_prog: str):
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    affine = Affine(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)
    memref = MemRef(ctx)
    linalg = Linalg(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()
    module.verify()

    printer = Printer()
    printer.print_op(module)


def test_linalg_generic_matmul():
    test_prog = """
module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {
  ^1(%A: !memref<[32, 32], !f32>, %B: !memref<[32, 32], !f32>, %C: !memref<[32, 32], !f32>):
    linalg.generic [
        "doc" = "C(d0, d1) += A(d0, d2) * B(d2, d1)",
        "indexing_maps" = [
          !affine_map<(d0, d1, d2)[] -> (d0, d2)>, 
          !affine_map<(d0, d1, d2)[] -> (d2, d1)>,
          !affine_map<(d0, d1, d2)[] -> (d0, d1)>],
        "library_call" = "linalg_matmul",
        "iterator_types" = ["parallel", "parallel", "reduction"]]
        ins(%A: !memref<[32, 32], !f32>, %B: !memref<[32, 32], !f32>) 
        outs(%C: !memref<[32, 32], !f32>) {
          ^2(%a: !f32, %b: !f32, %c: !f32):
            %d : !f32 = arith.mulf(%a : !f32, %b: !f32)
            %e : !f32 = arith.addf(%c : !f32, %d: !f32)
            linalg.yield(%e : !f32)
        }
  }
  std.call() ["callee" = !flat_symbol_ref<"test">]
}
    """
    parse_and_verify(test_prog)


if __name__ == "__main__":
    test_linalg_generic_matmul()