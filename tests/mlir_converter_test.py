import pytest

docutils = pytest.importorskip("xdsl.mlir")
from xdsl.mlir_converter import *
from xdsl.dialects.scf import Scf
from xdsl.dialects.func import Func
from xdsl.dialects.memref import MemRef
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.parser import Parser


def convert_and_verify(test_prog: str):
    ctx = MLContext()
    builtin = Builtin(ctx)
    func = Func(ctx)
    affine = Affine(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)
    memref = MemRef(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()
    module.verify()

    converter = MLIRConverter(ctx)
    with mlir.ir.Context() as mlir_ctx:
        mlir_ctx.allow_unregistered_dialects = True
        with mlir.ir.Location.unknown(mlir_ctx):
            mlir_prog = converter.convert_op(module)
            print(mlir_prog)
            assert (mlir_prog.verify())


def test_func_conversion():
    test_prog = """
module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    func.return()
  }
  func.call() ["callee" = !flat_symbol_ref<"test">]
}
    """
    convert_and_verify(test_prog)


def test_arith_conversion():
    test_prog = """
module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]{
  ^0(%arg : !i32):
    %0 : !i32 = arith.constant() ["value" = 0 : !i32]
    %res : !i32 = arith.addi(%arg : !i32, %0 : !i32) 
    func.return(%res : !i32)
  }
}
    """
    convert_and_verify(test_prog)


def test_scf_conversion():
    test_prog = """
module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]{
  ^0(%arg : !i32):
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %t : !i1 = arith.cmpi(%arg : !i32, %0 : !i32) ["predicate" = 1 : !i64]
    %res : !i32 = scf.if(%t : !i1) {
        %then : !i32 = arith.addi(%arg : !i32, %0 : !i32) 
        scf.yield(%then : !i32)
    } {
        %else : !i32 = arith.subi(%arg : !i32, %0 : !i32)
        scf.yield(%else : !i32)
    }
    func.return(%res : !i32)
  }
}
    """
    convert_and_verify(test_prog)


def test_variadic_conversion():
    test_prog = """
module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %1 : !memref<[1 : !i64], !i32> = memref.alloca() ["alignment" = 0 : !i64, "operand_segment_sizes" = !dense<!vector<[2 : !i64], !i32>, [0 : !i32, 0 : !i32]>]

    func.return()
  }
}
    """
    convert_and_verify(test_prog)


def test_memref_conversion():
    test_prog = """
module() {
  func.func() ["sym_name" = "sum", "function_type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "public"]{
  ^0(%0 : !i32, %1 : !i32):
    %2 : !index = arith.constant() ["value" = 0 : !index]
    %3 : !memref<[1 : !index], !i32> = memref.alloca() ["alignment" = 0 : !i64, "operand_segment_sizes" = !dense<!vector<[2 : !index], !i32>, [0 : !i32, 0 : !i32]>]
    memref.store(%0 : !i32, %3 : !memref<[1 : !index], !i32>, %2 : !index)
    %4 : !index = arith.constant() ["value" = 0 : !index]
    %5 : !memref<[1 : !index], !i32> = memref.alloca() ["alignment" = 0 : !i64, "operand_segment_sizes" = !dense<!vector<[2 : !index], !i32>, [0 : !i32, 0 : !i32]>]
    memref.store(%1 : !i32, %5 : !memref<[1 : !index], !i32>, %4 : !index)
    %6 : !index = arith.constant() ["value" = 0 : !index]
    %7 : !i32 = memref.load(%3 : !memref<[1 : !index], !i32>, %6 : !index)
    %8 : !index = arith.constant() ["value" = 0 : !index]
    %9 : !i32 = memref.load(%5 : !memref<[1 : !index], !i32>, %8 : !index)
    %10 : !i32 = arith.addi(%7 : !i32, %9 : !i32)
    func.return(%10 : !i32)
  }
}
    """
    convert_and_verify(test_prog)


# NOTE we currently do not test affine conversion because the dialect is incomplete
