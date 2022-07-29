from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil.stencil as stencil
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
from xdsl.dialects.stencil.stencil_inlining import InlineProducer, RerouteUse
import difflib


def parse(program: str):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    stencil.Stencil(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()

    printer = Printer()
    printer.print_op(module)


def apply_strategy_and_compare(program: str, expected_program: str,
                               strategy: Strategy):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    stencil.Stencil(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()
    imm_module: IOp = get_immutable_copy(module)

    rr = strategy.apply(imm_module)
    assert rr.isSuccess()

    # for debugging
    printer = Printer()
    print(f'Result after applying "{strategy}":')
    printer.print_op(rr.result_op.get_mutable_copy())
    print()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(rr.result_op.get_mutable_copy())

    diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                    expected_program.splitlines(True))
    if file.getvalue().strip() != expected_program.strip():
        print("Did not get expected output! Diff:")
        print(''.join(diff))
        assert False


def test_inlining_simple():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70,70,70]>, %arg1 : !stencil.field<[70,70,70]>):
    %1 : !stencil.temp<[66,66,63]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [66, 66, 63]]
    %2 : !stencil.temp<[64,64,60]>  = stencil.apply(%1 : !stencil.temp<[66,66,63]>) ["lb" = [1, 2, 3], "ub" = [65, 66, 63]] {
        ^bb0(%arg2: !stencil.temp<[66,66,63]>): 
        %3 : !f64 = stencil.access(%arg2: !stencil.temp<[66,66,63]>) ["offset" = [-1, 0, 0]]
        %4 : !f64 = stencil.access(%arg2: !stencil.temp<[66,66,63]>) ["offset" = [1, 0, 0]]
        %5 : !f64 = arith.addf(%3: !f64, %4: !f64)
        %6 : !stencil.result<!f64> = stencil.store_result(%5: !f64)
        stencil.return(%6: !stencil.result<!f64>)
    }
    %7 : !stencil.temp<[64,64,60]> = stencil.apply(%1 : !stencil.temp<[66,66,63]>, %2 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^bb0(%arg3: !stencil.temp<[66,66,63]>, %arg4: !stencil.temp<[64,64,60]>):
        %8 : !f64 = stencil.access(%arg3: !stencil.temp<[66,66,63]>) ["offset" = [0, 0, 0]]
        %9 : !f64 = stencil.access(%arg4: !stencil.temp<[64,64,60]>) ["offset" = [1, 2, 3]]
        %10 : !f64 = arith.addf(%8: !f64, %9: !f64)
        %11 : !stencil.result<!f64> = stencil.store_result(%10: !f64)
        stencil.return(%11: !stencil.result<!f64>)
    }
    stencil.store(%7 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [66 : !i64, 66 : !i64, 63 : !i64]]
  %3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%2 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>):
    %5 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %6 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 2 : !i64, 3 : !i64]]
    %7 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [2 : !i64, 2 : !i64, 3 : !i64]]
    %8 : !f64 = arith.addf(%6 : !f64, %7 : !f64)
    %9 : !f64 = arith.addf(%5 : !f64, %8 : !f64)
    %10 : !stencil.result<!f64> = stencil.store_result(%9 : !f64)
    stencil.return(%10 : !stencil.result<!f64>)
  }
  stencil.store(%3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    #  %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %2 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [66, 66, 63]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x66x63xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<66x66x63xf64>):  // no predecessors

    #       %5 = "stencil.access"(%arg2) {offset = [-1, 0, 0]} : (!stencil.temp<66x66x63xf64>) -> f64
    #       %6 = "stencil.access"(%arg2) {offset = [1, 0, 0]} : (!stencil.temp<66x66x63xf64>) -> f64
    #       %7 = "std.addf"(%5, %6) : (f64, f64) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #     }) {lb = [1, 2, 3], ub = [65, 66, 63]} : (!stencil.temp<66x66x63xf64>) -> !stencil.temp<64x64x60xf64>
    #     %4 = "stencil.apply"(%2, %3) ( {
    #     ^bb0(%arg2: !stencil.temp<66x66x63xf64>, %arg3: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %5 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<66x66x63xf64>) -> f64
    #       %6 = "stencil.access"(%arg3) {offset = [1, 2, 3]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %7 = "std.addf"(%5, %6) : (f64, f64) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x66x63xf64>, !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%4, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   })

    # Source after:
    # //   func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.load %0([0, 0, 0] : [66, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x66x63xf64>
    # //     %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %4 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<66x66x63xf64>) -> f64
    # //       %5 = stencil.access %arg2 [0, 2, 3] : (!stencil.temp<66x66x63xf64>) -> f64
    # //       %6 = stencil.access %arg2 [2, 2, 3] : (!stencil.temp<66x66x63xf64>) -> f64
    # //       %7 = addf %5, %6 : f64
    # //       %8 = addf %4, %7 : f64
    # //       %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    # //       stencil.return %9 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(
        before, after,
        seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_inlining_simple_index():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !f64, %arg1 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[64,64,60]>  = stencil.apply(%arg0 : !f64) ["lb" = [1, 2, 3], "ub" = [65, 66, 63]] {
        ^0(%arg2: !f64): 
        %2 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [2, -1, 1]]
        %3 : !index = arith.constant() ["value" = 20 : !index]
        %4 : !f64 = arith.constant() ["value" = 0 : !i32]
        %5 : !i1 = arith.cmpi(%2 : !index, %3 : !index) ["predicate" = 2 : !i64]
        %6 : !f64 = arith.select(%5 : !i1, %arg2 : !f64, %4: !f64)
        %7 : !stencil.result<!f64> = stencil.store_result(%6 : !f64)
        stencil.return(%7 : !stencil.result<!f64>)
    }
    %8 : !stencil.temp<[64,64,60]> = stencil.apply(%arg0 : !f64, %1 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg3: !f64, %arg4: !stencil.temp<[64,64,60]>):
        %9 : !i64 = stencil.access(%arg4: !stencil.temp<[64,64,60]>) ["offset" = [1, 2, 3]]
        %10 : !f64 = arith.addf(%9: !i64, %arg3: !f64)
        %11 : !stencil.result<!f64> = stencil.store_result(%10: !f64)
        stencil.return(%11 : !stencil.result<!f64>)
    }
    stencil.store(%8: !stencil.temp<[64, 64, 60]>, %arg1: !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !f64, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%0 : !f64) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%3 : !f64):
    %4 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [3 : !i64, 1 : !i64, 4 : !i64]]
    %5 : !index = arith.constant() ["value" = 20 : !index]
    %6 : !f64 = arith.constant() ["value" = 0 : !i32]
    %7 : !i1 = arith.cmpi(%4 : !index, %5 : !index) ["predicate" = 2 : !i64]
    %8 : !f64 = arith.select(%7 : !i1, %3 : !f64, %6 : !f64)
    %9 : !f64 = arith.addf(%8 : !f64, %3 : !f64)
    %10 : !stencil.result<!f64> = stencil.store_result(%9 : !f64)
    stencil.return(%10 : !stencil.result<!f64>)
  }
  stencil.store(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    # "func"() ( {
    #   ^bb0(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.apply"(%arg0) ( {
    #     ^bb0(%arg2: f64):  // no predecessors
    #       %3 = "stencil.index"() {dim = 2 : i64, offset = [2, -1, 1]} : () -> index
    #       %c20 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %4 = "std.cmpi"(%3, %c20) {predicate = 2 : i64} : (index, index) -> i1
    #       %5 = "std.select"(%4, %arg2, %cst) : (i1, f64, f64) -> f64
    #       %6 = "stencil.store_result"(%5) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%6) : (!stencil.result<f64>) -> ()
    #     }) {lb = [1, 2, 3], ub = [65, 66, 63]} : (f64) -> !stencil.temp<64x64x60xf64>
    #     %2 = "stencil.apply"(%arg0, %1) ( {
    #     ^bb0(%arg2: f64, %arg3: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %3 = "stencil.access"(%arg3) {offset = [1, 2, 3]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %4 = "std.addf"(%3, %arg2) : (f64, f64) -> f64
    #       %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%5) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (f64, !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%2, %0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "simple_index", type = (f64, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # Source after:
    # // module  {
    # //   func @simple_index(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<64x64x60xf64> {
    # //       %c20 = constant 20 : index
    # //       %cst = constant 0.000000e+00 : f64
    # //       %2 = stencil.index 2 [3, 1, 4] : index
    # //       %3 = cmpi slt, %2, %c20 : index
    # //       %4 = select %3, %arg2, %cst : f64
    # //       %5 = addf %4, %arg2 : f64
    # //       %6 = stencil.store_result %5 : (f64) -> !stencil.result<f64>
    # //       stencil.return %6 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %1 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(
        before, after,
        seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_inlining_simple_ifelse():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !f64, %arg1 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[64,64,60]> = stencil.apply(%arg0 : !f64) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^0(%arg2: !f64): 
        %true : !i1 = arith.constant() ["value" = 1 : !i1]
        %2 : !stencil.result<!f64> = scf.if(%true : !i1) {
            %3 : !stencil.result<!f64> = stencil.store_result(%arg2 : !f64)
            scf.yield(%3 : !stencil.result<!f64>)
        } {
            %4 : !stencil.result<!f64> = stencil.store_result(%arg2 : !f64)
            scf.yield(%4 : !stencil.result<!f64>)
        }
        stencil.return(%2 : !stencil.result<!f64>)
    }
    %5 : !stencil.temp<[64,64,60]> = stencil.apply(%1 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg3: !stencil.temp<[64,64,60]>):
        %6 : !f64 = stencil.access(%arg3: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %7 : !stencil.result<!f64> = stencil.store_result(%6 : !f64)
        stencil.return(%7 : !stencil.result<!f64>)
    }
    stencil.store(%5 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !f64, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%0 : !f64) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%3 : !f64):
    %4 : !i1 = arith.constant() ["value" = 1 : !i1]
    %5 : !f64 = scf.if(%4 : !i1) {
      scf.yield(%3 : !f64)
    } {
      scf.yield(%3 : !f64)
    }
    %6 : !stencil.result<!f64> = stencil.store_result(%5 : !f64)
    stencil.return(%6 : !stencil.result<!f64>)
  }
  stencil.store(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    #   "func"() ( {
    #   ^bb0(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.apply"(%arg0) ( {
    #     ^bb0(%arg2: f64):  // no predecessors
    #       %true = "std.constant"() {value = true} : () -> i1
    #       %3 = "scf.if"(%true) ( {
    #         %4 = "stencil.store_result"(%arg2) : (f64) -> !stencil.result<f64>
    #         "scf.yield"(%4) : (!stencil.result<f64>) -> ()
    #       },  {
    #         %4 = "stencil.store_result"(%arg2) : (f64) -> !stencil.result<f64>
    #         "scf.yield"(%4) : (!stencil.result<f64>) -> ()
    #       }) : (i1) -> !stencil.result<f64>
    #       "stencil.return"(%3) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (f64) -> !stencil.temp<64x64x60xf64>
    #     %2 = "stencil.apply"(%1) ( {
    #     ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %3 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %4 = "stencil.store_result"(%3) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%4) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%2, %0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "simple_ifelse", type = (f64, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # Source after:
    # // module  {
    # //   func @simple_ifelse(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<64x64x60xf64> {
    # //       %true = constant true
    # //       %2 = scf.if %true -> (f64) {
    # //         scf.yield %arg2 : f64
    # //       } else {
    # //         scf.yield %arg2 : f64
    # //       }
    # //       %3 = stencil.store_result %2 : (f64) -> !stencil.result<f64>
    # //       stencil.return %3 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %1 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(
        before, after,
        seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_inlining_multiple_edges():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70, 70, 70]>, %arg1 : !stencil.field<[70, 70, 70]>, %arg2 : !stencil.field<[70, 70, 70]>):
    %0 : !stencil.temp<[66,64,60]> = stencil.load(%arg0 : !stencil.field<[70, 70, 70]>) ["lb" = [-1, 0, 0], "ub" = [65, 64, 60]]
    (%1 : !stencil.temp<[64,64,60]>, %2 : !stencil.temp<[64,64,60]>)  = stencil.apply(%0 : !stencil.temp<[66,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^0(%arg3 : !stencil.temp<[66,64,60]>): 
          %3 : !f64 = stencil.access(%arg3 : !stencil.temp<[66,64,60]>) ["offset" = [-1, 0, 0]]
          %4 : !f64 = stencil.access(%arg3 : !stencil.temp<[66,64,60]>) ["offset" = [1, 0, 0]]
          %5 : !stencil.result<!f64> = stencil.store_result(%3 : !f64)
          %6 : !stencil.result<!f64> = stencil.store_result(%4 : !f64)
          stencil.return(%5 : !stencil.result<!f64>, %6 : !stencil.result<!f64>)
    }
    %7 : !stencil.temp<[64,64,60]> = stencil.load(%arg1 : !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    %8 : !stencil.temp<[64,64,60]> = stencil.apply(%0 : !stencil.temp<[66,64,60]>, %1 : !stencil.temp<[64,64,60]>, %2 : !stencil.temp<[64,64,60]>, %7 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg4 : !stencil.temp<[66,64,60]>, %arg5 : !stencil.temp<[64,64,60]>, %arg6 : !stencil.temp<[64,64,60]>, %arg7 : !stencil.temp<[64,64,60]>):
        %9 : !f64 = stencil.access(%arg4: !stencil.temp<[66,64,60]>) ["offset" = [0, 0, 0]]
        %10 : !f64 = stencil.access(%arg5: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %11 : !f64 = stencil.access(%arg6: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %12 : !f64 = stencil.access(%arg7: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %13 : !f64 = arith.addf(%9: !f64, %10: !f64)
        %14 : !f64 = arith.addf(%11: !f64, %12: !f64)
        %15 : !f64 = arith.addf(%13: !f64, %14: !f64)
        %16 : !stencil.result<!f64> = stencil.store_result(%15: !f64)
        stencil.return(%16 : !stencil.result<!f64>)
    }
    stencil.store(%8: !stencil.temp<[64, 64, 60]>, %arg2: !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %3 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [-1 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 64 : !i64, 60 : !i64]]
  %4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.load(%1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  %5 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%3 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>, %4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%6 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>, %7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %8 : !f64 = stencil.access(%6 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %9 : !f64 = stencil.access(%6 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [-1 : !i64, 0 : !i64, 0 : !i64]]
    %10 : !f64 = stencil.access(%6 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [1 : !i64, 0 : !i64, 0 : !i64]]
    %11 : !f64 = stencil.access(%7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %12 : !f64 = arith.addf(%8 : !f64, %9 : !f64)
    %13 : !f64 = arith.addf(%10 : !f64, %11 : !f64)
    %14 : !f64 = arith.addf(%12 : !f64, %13 : !f64)
    %15 : !stencil.result<!f64> = stencil.store_result(%14 : !f64)
    stencil.return(%15 : !stencil.result<!f64>)
  }
  stencil.store(%5 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    # "func"() ( {
    # ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>):  // no predecessors
    #   %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %2 = "stencil.cast"(%arg2) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %3 = "stencil.load"(%0) {lb = [-1, 0, 0], ub = [65, 64, 60]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
    #   %4:2 = "stencil.apply"(%3) ( {
    #   ^bb0(%arg3: !stencil.temp<66x64x60xf64>):  // no predecessors
    #     %7 = "stencil.access"(%arg3) {offset = [-1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %8 = "stencil.access"(%arg3) {offset = [1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %9 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #     %10 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%9, %10) : (!stencil.result<f64>, !stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x64x60xf64>) -> (!stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>)
    #   %5 = "stencil.load"(%1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
    #   %6 = "stencil.apply"(%3, %4#0, %4#1, %5) ( {
    #   ^bb0(%arg3: !stencil.temp<66x64x60xf64>, %arg4: !stencil.temp<64x64x60xf64>, %arg5: !stencil.temp<64x64x60xf64>, %arg6: !stencil.temp<64x64x60xf64>):  // no predecessors
    #     %7 = "stencil.access"(%arg3) {offset = [0, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %8 = "stencil.access"(%arg4) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #     %9 = "stencil.access"(%arg5) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #     %10 = "stencil.access"(%arg6) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #     %11 = "std.addf"(%7, %8) : (f64, f64) -> f64
    #     %12 = "std.addf"(%9, %10) : (f64, f64) -> f64
    #     %13 = "std.addf"(%11, %12) : (f64, f64) -> f64
    #     %14 = "stencil.store_result"(%13) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%14) : (!stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x64x60xf64>, !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #   "stencil.store"(%6, %2) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #   "std.return"() : () -> ()
    # }) {stencil.program, sym_name = "multiple_edges", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # Source after:
    # // module  {
    # //   func @multiple_edges(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %3 = stencil.load %0([-1, 0, 0] : [65, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
    # //     %4 = stencil.load %1([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
    # //     %5 = stencil.apply (%arg3 = %3 : !stencil.temp<66x64x60xf64>, %arg4 = %4 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    # //       %7 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    # //       %8 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    # //       %9 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    # //       %10 = addf %6, %7 : f64
    # //       %11 = addf %8, %9 : f64
    # //       %12 = addf %10, %11 : f64
    # //       %13 = stencil.store_result %12 : (f64) -> !stencil.result<f64>
    # //       stencil.return %13 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %5 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(
        before, after,
        seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_inlining_reroute():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70, 70, 70]>, %arg1 : !stencil.field<[70, 70, 70]>, %arg2 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[67,66,63]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [-1, 0, 0], "ub" = [66, 66, 63]]
    %2 : !stencil.temp<[65,66,63]>  = stencil.apply(%1 : !stencil.temp<[67,66,63]>) ["lb" = [0, 0, 0], "ub" = [65, 66, 63]] {
        ^1(%arg4: !stencil.temp<[67,66,63]>): 
        %3 : !f64 = stencil.access(%arg4: !stencil.temp<[67,66,63]>) ["offset" = [-1, 0, 0]]
        %4 : !f64 = stencil.access(%arg4: !stencil.temp<[67,66,63]>) ["offset" = [1, 0, 0]]
        %5 : !f64 = arith.addf(%3: !f64, %4: !f64)
        %6 : !stencil.result<!f64> = stencil.store_result(%5: !f64)
        stencil.return(%6: !stencil.result<!f64>)
    }
    %7 : !stencil.temp<[64,64,60]>  = stencil.apply(%2 : !stencil.temp<[65,66,63]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg5: !stencil.temp<[65,66,63]>): 
        %7 : !f64 = stencil.access(%arg5: !stencil.temp<[65,66,63]>) ["offset" = [0, 0, 0]]
        %8 : !f64 = stencil.access(%arg5: !stencil.temp<[65,66,63]>) ["offset" = [1, 2, 3]]
        %9 : !f64 = arith.addf(%7: !f64, %8: !f64)
        %10 : !stencil.result<!f64> = stencil.store_result(%9: !f64)
        stencil.return(%10: !stencil.result<!f64>)
    }
    stencil.store(%2: !stencil.temp<[65,66,63]>, %arg1: !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [65, 66, 63]]
    stencil.store(%7: !stencil.temp<[64,64,60]>, %arg2: !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    intermediate  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %3 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [-1 : !i64, 0 : !i64, 0 : !i64], "ub" = [66 : !i64, 66 : !i64, 63 : !i64]]
  %4 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]> = stencil.apply(%3 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]] {
  ^1(%5 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>):
    %6 : !f64 = stencil.access(%5 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [-1 : !i64, 0 : !i64, 0 : !i64]]
    %7 : !f64 = stencil.access(%5 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [1 : !i64, 0 : !i64, 0 : !i64]]
    %8 : !f64 = arith.addf(%6 : !f64, %7 : !f64)
    %9 : !stencil.result<!f64> = stencil.store_result(%8 : !f64)
    stencil.return(%9 : !stencil.result<!f64>)
  }
  (%10 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %11 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) = stencil.apply(%4 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %4 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]] {
  ^2(%12 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %13 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>):
    %14 : !f64 = stencil.access(%12 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %15 : !f64 = stencil.access(%12 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [1 : !i64, 2 : !i64, 3 : !i64]]
    %16 : !f64 = arith.addf(%14 : !f64, %15 : !f64)
    %17 : !stencil.result<!f64> = stencil.store_result(%16 : !f64)
    %18 : !f64 = stencil.access(%13 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %19 : !stencil.result<!f64> = stencil.store_result(%18 : !f64)
    stencil.return(%17 : !stencil.result<!f64>, %19 : !stencil.result<!f64>)
  }
  stencil.store(%11 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]]
  stencil.store(%10 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %3 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [-1 : !i64, 0 : !i64, 0 : !i64], "ub" = [66 : !i64, 66 : !i64, 63 : !i64]]
  (%4 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %5 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) = stencil.apply(%3 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]] {
  ^1(%6 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>):
    %7 : !f64 = stencil.access(%6 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [-1 : !i64, 0 : !i64, 0 : !i64]]
    %8 : !f64 = stencil.access(%6 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [1 : !i64, 0 : !i64, 0 : !i64]]
    %9 : !f64 = arith.addf(%7 : !f64, %8 : !f64)
    %10 : !f64 = stencil.access(%6 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 2 : !i64, 3 : !i64]]
    %11 : !f64 = stencil.access(%6 : !stencil.temp<[67 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [2 : !i64, 2 : !i64, 3 : !i64]]
    %12 : !f64 = arith.addf(%10 : !f64, %11 : !f64)
    %13 : !f64 = arith.addf(%9 : !f64, %12 : !f64)
    %14 : !stencil.result<!f64> = stencil.store_result(%13 : !f64)
    %15 : !stencil.result<!f64> = stencil.store_result(%9 : !f64)
    stencil.return(%14 : !stencil.result<!f64>, %15 : !stencil.result<!f64>)
  }
  stencil.store(%5 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]]
  stencil.store(%4 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    # Source before:
    #   "func"() ( {
    #   ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %2 = "stencil.cast"(%arg2) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %3 = "stencil.load"(%0) {lb = [-1, 0, 0], ub = [66, 66, 63]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<67x66x63xf64>
    #     %4 = "stencil.apply"(%3) ( {
    #     ^bb0(%arg3: !stencil.temp<67x66x63xf64>):  // no predecessors
    #       %6 = "stencil.access"(%arg3) {offset = [-1, 0, 0]} : (!stencil.temp<67x66x63xf64>) -> f64
    #       %7 = "stencil.access"(%arg3) {offset = [1, 0, 0]} : (!stencil.temp<67x66x63xf64>) -> f64
    #       %8 = "std.addf"(%6, %7) : (f64, f64) -> f64
    #       %9 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%9) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [65, 66, 63]} : (!stencil.temp<67x66x63xf64>) -> !stencil.temp<65x66x63xf64>
    #     %5 = "stencil.apply"(%4) ( {
    #     ^bb0(%arg3: !stencil.temp<65x66x63xf64>):  // no predecessors
    #       %6 = "stencil.access"(%arg3) {offset = [0, 0, 0]} : (!stencil.temp<65x66x63xf64>) -> f64
    #       %7 = "stencil.access"(%arg3) {offset = [1, 2, 3]} : (!stencil.temp<65x66x63xf64>) -> f64
    #       %8 = "std.addf"(%6, %7) : (f64, f64) -> f64
    #       %9 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%9) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%4, %1) {lb = [0, 0, 0], ub = [65, 66, 63]} : (!stencil.temp<65x66x63xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "stencil.store"(%5, %2) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "reroute", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # Source intermediate:
    #   // module  {
    #   //   func @reroute(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    #   //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   //     %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   //     %3 = stencil.load %0([-1, 0, 0] : [66, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<67x66x63xf64>
    #   //     %4 = stencil.apply (%arg3 = %3 : !stencil.temp<67x66x63xf64>) -> !stencil.temp<65x66x63xf64> {
    #   //       %6 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    #   //       %7 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    #   //       %8 = addf %6, %7 : f64
    #   //       %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    #   //       stencil.return %9 : !stencil.result<f64>
    #   //     } to ([0, 0, 0] : [65, 66, 63])
    #   //     %5:2 = stencil.apply (%arg3 = %4 : !stencil.temp<65x66x63xf64>, %arg4 = %4 : !stencil.temp<65x66x63xf64>) -> (!stencil.temp<65x66x63xf64>, !stencil.temp<65x66x63xf64>) {
    #   //       %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<65x66x63xf64>) -> f64
    #   //       %7 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<65x66x63xf64>) -> f64
    #   //       %8 = addf %6, %7 : f64
    #   //       %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    #   //       %10 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<65x66x63xf64>) -> f64
    #   //       %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    #   //       stencil.return %9, %11 : !stencil.result<f64>, !stencil.result<f64>
    #   //     } to ([0, 0, 0] : [65, 66, 63])
    #   //     stencil.store %5#1 to %1([0, 0, 0] : [65, 66, 63]) : !stencil.temp<65x66x63xf64> to !stencil.field<70x70x70xf64>
    #   //     stencil.store %5#0 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<65x66x63xf64> to !stencil.field<70x70x70xf64>
    #   //     return
    #   //   }
    #   // }

    # Source after:

    #   // module  {
    #   //   func @reroute(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    #   //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   //     %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   //     %3 = stencil.load %0([-1, 0, 0] : [66, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<67x66x63xf64>
    #   //     %4:2 = stencil.apply (%arg3 = %3 : !stencil.temp<67x66x63xf64>) -> (!stencil.temp<65x66x63xf64>, !stencil.temp<65x66x63xf64>) {
    #   //       %5 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    #   //       %6 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    #   //       %7 = addf %5, %6 : f64
    #   //       %8 = stencil.access %arg3 [0, 2, 3] : (!stencil.temp<67x66x63xf64>) -> f64
    #   //       %9 = stencil.access %arg3 [2, 2, 3] : (!stencil.temp<67x66x63xf64>) -> f64
    #   //       %10 = addf %8, %9 : f64
    #   //       %11 = addf %7, %10 : f64
    #   //       %12 = stencil.store_result %11 : (f64) -> !stencil.result<f64>
    #   //       %13 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    #   //       stencil.return %12, %13 : !stencil.result<f64>, !stencil.result<f64>
    #   //     } to ([0, 0, 0] : [65, 66, 63])
    #   //     stencil.store %4#1 to %1([0, 0, 0] : [65, 66, 63]) : !stencil.temp<65x66x63xf64> to !stencil.field<70x70x70xf64>
    #   //     stencil.store %4#0 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<65x66x63xf64> to !stencil.field<70x70x70xf64>
    #   //     return
    #   //   }
    #   // }

    parse(before)
    parse(intermediate)
    parse(after)

    # Only performing the rerouting step
    apply_strategy_and_compare(
        before, intermediate,
        multiRoot(
            matchTopToBottom(RerouteUse.Match()), lambda matched_consumer:
            topToBottom(RerouteUse(*matched_consumer))))

    # Only performing the inlining step
    apply_strategy_and_compare(
        intermediate, after,
        topToBottom(InlineProducer()) ^ topToBottom(GarbageCollect()))

    # combining both steps
    apply_strategy_and_compare(
        before, after,
        multiRoot(
            matchTopToBottom(RerouteUse.Match()), lambda matched_consumer:
            topToBottom(RerouteUse(*matched_consumer)))
        ^ topToBottom(InlineProducer()) ^ topToBottom(GarbageCollect()))


def test_inlining_avoid_redundant():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70, 70, 70]>, %arg1 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[66,64,60]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [-1, 0, 0], "ub" = [65, 64, 60]]
    %2 : !stencil.temp<[64,64,60]> = stencil.apply(%1 : !stencil.temp<[66,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg2: !stencil.temp<[66,64,60]>): 
        %3 : !f64 = stencil.access(%arg2: !stencil.temp<[66,64,60]>) ["offset" = [-1, 0, 0]]
        %4 : !f64 = stencil.access(%arg2: !stencil.temp<[66,64,60]>) ["offset" = [1, 0, 0]]
        %5 : !f64 = arith.addf(%3: !f64, %4: !f64)
        %6 : !stencil.result<!f64> = stencil.store_result(%5: !f64)
        stencil.return(%6: !stencil.result<!f64>)
    }
    %7 : !stencil.temp<[64,64,60]> = stencil.apply(%2 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg3: !stencil.temp<[64,64,60]>): 
        %8 : !f64 = stencil.access(%arg3: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %9 : !f64 = stencil.access(%arg3: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %10 : !f64 = arith.addf(%8: !f64, %9: !f64)
        %11 : !stencil.result<!f64> = stencil.store_result(%10: !f64)
        stencil.return(%11: !stencil.result<!f64>)
    }
    stencil.store(%7: !stencil.temp<[64,64,60]>, %arg1: !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [-1 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 64 : !i64, 60 : !i64]]
  %3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%2 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%4 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>):
    %5 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [-1 : !i64, 0 : !i64, 0 : !i64]]
    %6 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [1 : !i64, 0 : !i64, 0 : !i64]]
    %7 : !f64 = arith.addf(%5 : !f64, %6 : !f64)
    %8 : !f64 = arith.addf(%7 : !f64, %7 : !f64)
    %9 : !stencil.result<!f64> = stencil.store_result(%8 : !f64)
    stencil.return(%9 : !stencil.result<!f64>)
  }
  stencil.store(%3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:

    # "func"() ( {
    # ^bb0(%arg0: !stencil.field<? x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #   %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %2 = "stencil.load"(%0) {lb = [-1, 0, 0], ub = [65, 64, 60]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
    #   %3 = "stencil.apply"(%2) ( {
    #   ^bb0(%arg2: !stencil.temp<66x64x60xf64>):  // no predecessors
    #     %5 = "stencil.access"(%arg2) {offset = [-1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %6 = "stencil.access"(%arg2) {offset = [1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %7 = "std.addf"(%5, %6) : (f64, f64) -> f64
    #     %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #   %4 = "stencil.apply"(%3) ( {
    #   ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #     %5 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #     %6 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #     %7 = "std.addf"(%5, %6) : (f64, f64) -> f64
    #     %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #   "stencil.store"(%4, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #   "std.return"() : () -> ()
    # }) {stencil.program, sym_name = "avoid_redundant", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # Source after:
    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %2 = "stencil.load"(%0) {lb = [-1, 0, 0], ub = [65, 64, 60]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<66x64x60xf64>):  // no predecessors
    #       %4 = "stencil.access"(%arg2) {offset = [-1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #       %5 = "stencil.access"(%arg2) {offset = [1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #       %6 = "std.addf"(%4, %5) : (f64, f64) -> f64
    #       %7 = "std.addf"(%6, %6) : (f64, f64) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%3, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "avoid_redundant", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()

    parse(before)
    parse(after)

    apply_strategy_and_compare(
        before, after,
        seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_inlining_root():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70, 70, 70]>, %arg1 : !stencil.field<[70, 70, 70]>, %arg2 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[65,66,63]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [65, 66, 63]]
    %2 : !stencil.temp<[64,64,60]>  = stencil.apply(%1 : !stencil.temp<[65,66,63]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg4: !stencil.temp<[65,66,63]>):
        %3 : !f64 = stencil.access(%arg4: !stencil.temp<[65,66,63]>) ["offset" = [0, 0, 0]]
        %4 : !stencil.result<!f64> = stencil.store_result(%3: !f64)
        stencil.return(%4: !stencil.result<!f64>)
    }
    %5 : !stencil.temp<[64,64,60]> = stencil.apply(%1 : !stencil.temp<[65,66,63]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg5: !stencil.temp<[65,66,63]>):
        %6 : !f64 = stencil.access(%arg5: !stencil.temp<[65,66,63]>) ["offset" = [1, 2, 3]]
        %7 : !stencil.result<!f64> = stencil.store_result(%6: !f64)
        stencil.return(%7: !stencil.result<!f64>)
    }
    stencil.store(%2: !stencil.temp<[64,64,60]>, %arg1: !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    stencil.store(%5: !stencil.temp<[64,64,60]>, %arg2: !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    intermediate = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %3 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]]
  %4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%3 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%5 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>):
    %6 : !f64 = stencil.access(%5 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %7 : !stencil.result<!f64> = stencil.store_result(%6 : !f64)
    stencil.return(%7 : !stencil.result<!f64>)
  }
  (%8 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %9 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) = stencil.apply(%3 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^2(%10 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %11 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %12 : !f64 = stencil.access(%10 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [1 : !i64, 2 : !i64, 3 : !i64]]
    %13 : !stencil.result<!f64> = stencil.store_result(%12 : !f64)
    %14 : !f64 = stencil.access(%11 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %15 : !stencil.result<!f64> = stencil.store_result(%14 : !f64)
    stencil.return(%13 : !stencil.result<!f64>, %15 : !stencil.result<!f64>)
  }
  stencil.store(%8 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  stencil.store(%9 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %3 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [65 : !i64, 66 : !i64, 63 : !i64]]
  (%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %5 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) = stencil.apply(%3 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%6 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>, %7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %8 : !f64 = stencil.access(%6 : !stencil.temp<[65 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [1 : !i64, 2 : !i64, 3 : !i64]]
    %9 : !stencil.result<!f64> = stencil.store_result(%8 : !f64)
    %10 : !f64 = stencil.access(%7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %11 : !stencil.result<!f64> = stencil.store_result(%10 : !f64)
    stencil.return(%9 : !stencil.result<!f64>, %11 : !stencil.result<!f64>)
  }
  stencil.store(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  stencil.store(%5 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %2 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # before:

    # "func"() ( {
    # ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>):  // no predecessors
    #   %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %2 = "stencil.cast"(%arg2) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %3 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [65, 66, 63]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<65x66x63xf64>
    #   %4 = "stencil.apply"(%3) ( {
    #   ^bb0(%arg3: !stencil.temp<65x66x63xf64>):  // no predecessors
    #     %6 = "stencil.access"(%arg3) {offset = [0, 0, 0]} : (!stencil.temp<65x66x63xf64>) -> f64
    #     %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%7) : (!stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64>
    #   %5 = "stencil.apply"(%3) ( {
    #   ^bb0(%arg3: !stencil.temp<65x66x63xf64>):  // no predecessors
    #     %6 = "stencil.access"(%arg3) {offset = [1, 2, 3]} : (!stencil.temp<65x66x63xf64>) -> f64
    #     %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%7) : (!stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64>
    #   "stencil.store"(%4, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #   "stencil.store"(%5, %2) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #   "std.return"() : () -> ()
    # }) {stencil.program, sym_name = "root", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # // intermediate result:

    # // module  {
    # //   func @root(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %3 = stencil.load %0([0, 0, 0] : [65, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<65x66x63xf64>
    # //     %4 = stencil.apply (%arg3 = %3 : !stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<65x66x63xf64>) -> f64
    # //       %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    # //       stencil.return %7 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     %5:2 = stencil.apply (%arg3 = %3 : !stencil.temp<65x66x63xf64>, %arg4 = %4 : !stencil.temp<64x64x60xf64>) -> (!stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>) {
    # //       %6 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<65x66x63xf64>) -> f64
    # //       %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    # //       %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    # //       %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    # //       stencil.return %7, %9 : !stencil.result<f64>, !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %5#1 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     stencil.store %5#0 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    # // final result:

    # // module  {
    # //   func @root(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %3 = stencil.load %0([0, 0, 0] : [65, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<65x66x63xf64>
    # //     %4:2 = stencil.apply (%arg3 = %3 : !stencil.temp<65x66x63xf64>) -> (!stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>) {
    # //       %5 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<65x66x63xf64>) -> f64
    # //       %6 = stencil.store_result %5 : (f64) -> !stencil.result<f64>
    # //       %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<65x66x63xf64>) -> f64
    # //       %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    # //       stencil.return %6, %8 : !stencil.result<f64>, !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %4#1 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     stencil.store %4#0 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    parse(before)
    parse(intermediate)
    parse(after)

    # TODO: to make this work we need the other matching step in the rerouting rewrite.
    # The actual rewrite should be the same, only the matching has to be improved

    # Only performing the rerouting step
    # apply_strategy_and_compare(
    #     before, intermediate,
    #     multiRoot(
    #         matchTopToBottom(RerouteUse.Match()), lambda matched_consumer:
    #         topToBottom(RerouteUse(*matched_consumer))))

    # # Only performing the inlining step
    # apply_strategy_and_compare(
    #     intermediate, after,
    #     topToBottom(InlineProducer()) ^ topToBottom(GarbageCollect()))

    # # combining both steps
    # apply_strategy_and_compare(
    #     before, after,
    #     multiRoot(
    #         matchTopToBottom(RerouteUse.Match()), lambda matched_consumer:
    #         topToBottom(RerouteUse(*matched_consumer)))
    #     ^ topToBottom(InlineProducer()) ^ topToBottom(GarbageCollect()))


def test_inlining_dyn_access():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70, 70, 70]>, %arg1 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[68,66,62]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [-2, -1, -2], "ub" = [66, 65, 60]]
    %2 : !stencil.temp<[68,66,62]>  = stencil.apply(%1 : !stencil.temp<[68,66,62]>) ["lb" = [-2, -1, -2], "ub" = [66, 65, 61]] {
        ^1(%arg2 : !stencil.temp<[68,66,62]>):
        %3 : !f64 = stencil.access(%arg2 : !stencil.temp<[68,66,62]>) ["offset" = [0, 0, -1]]
        %4 : !stencil.result<!f64> = stencil.store_result(%3: !f64)
        stencil.return(%4: !stencil.result<!f64>)
    }
    %5 : !stencil.temp<[66,64,60]> = stencil.apply(%2 : !stencil.temp<[68,66,62]>) ["lb" = [-1, 0, 0], "ub" = [65, 64, 60]] {
        ^2(%arg3 : !stencil.temp<[68,66,62]>):
        %6 : !index = stencil.index() ["dim" = 0 : !i64, "offset" = [0, 0, 0]]
        %7 : !f64 = stencil.dyn_access(%arg3 : !stencil.temp<[68,66,62]>, %6 : !index, %6 : !index, %6 : !index) ["lb" = [-1, -1, -1], "ub" = [1, 1, 1]]
        %8 : !stencil.result<!f64> = stencil.store_result(%7: !f64)
        stencil.return(%8: !stencil.result<!f64>)
    }
    %9 : !stencil.temp<[64, 64, 60]> = stencil.apply(%5 : !stencil.temp<[66,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
      ^3(%arg4 : !stencil.temp<[66,64,60]>):
        %10 : !f64 = stencil.access(%arg4 : !stencil.temp<[66,64,60]>) ["offset" = [-1, 0, 0]]
        %11 : !f64 = stencil.access(%arg4 : !stencil.temp<[66,64,60]>) ["offset" = [1, 0, 0]]
        %12 : !f64 = arith.addf(%10 : !f64, %11 : !f64)
        %13 : !stencil.result<!f64> = stencil.store_result(%12 : !f64)
        stencil.return(%13 : !stencil.result<!f64>)
    }
    stencil.store(%9: !stencil.temp<[64, 64, 60]>, %arg1: !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [-2 : !i64, -1 : !i64, -2 : !i64], "ub" = [66 : !i64, 65 : !i64, 60 : !i64]]
  %3 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]> = stencil.apply(%2 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>) ["lb" = [-2 : !i64, -1 : !i64, -2 : !i64], "ub" = [66 : !i64, 65 : !i64, 61 : !i64]] {
  ^1(%4 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>):
    %5 : !f64 = stencil.access(%4 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, -1 : !i64]]
    %6 : !stencil.result<!f64> = stencil.store_result(%5 : !f64)
    stencil.return(%6 : !stencil.result<!f64>)
  }
  %7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%3 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^2(%8 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>):
    %9 : !index = stencil.index() ["dim" = 0 : !i64, "offset" = [-1 : !i64, 0 : !i64, 0 : !i64]]
    %10 : !f64 = stencil.dyn_access(%8 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>, %9 : !index, %9 : !index, %9 : !index) ["lb" = [-2 : !i64, -1 : !i64, -1 : !i64], "ub" = [0 : !i64, 1 : !i64, 1 : !i64]]
    %11 : !index = stencil.index() ["dim" = 0 : !i64, "offset" = [1 : !i64, 0 : !i64, 0 : !i64]]
    %12 : !f64 = stencil.dyn_access(%8 : !stencil.temp<[68 : !i64, 66 : !i64, 62 : !i64]>, %11 : !index, %11 : !index, %11 : !index) ["lb" = [0 : !i64, -1 : !i64, -1 : !i64], "ub" = [2 : !i64, 1 : !i64, 1 : !i64]]
    %13 : !f64 = arith.addf(%10 : !f64, %12 : !f64)
    %14 : !stencil.result<!f64> = stencil.store_result(%13 : !f64)
    stencil.return(%14 : !stencil.result<!f64>)
  }
  stencil.store(%7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # before:

    # "func"() ( {
    # ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #   %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %2 = "stencil.load"(%0) {lb = [-2, -1, -2], ub = [66, 65, 60]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<68x66x62xf64>
    #   %3 = "stencil.apply"(%2) ( {
    #   ^bb0(%arg2: !stencil.temp<68x66x62xf64>):  // no predecessors
    #     %6 = "stencil.access"(%arg2) {offset = [0, 0, -1]} : (!stencil.temp<68x66x62xf64>) -> f64
    #     %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%7) : (!stencil.result<f64>) -> ()
    #   }) {lb = [-2, -1, -1], ub = [66, 65, 61]} : (!stencil.temp<68x66x62xf64>) -> !stencil.temp<68x66x62xf64>
    #   %4 = "stencil.apply"(%3) ( {
    #   ^bb0(%arg2: !stencil.temp<68x66x62xf64>):  // no predecessors
    #     %6 = "stencil.index"() {dim = 0 : i64, offset = [0, 0, 0]} : () -> index
    #     %7 = "stencil.dyn_access"(%arg2, %6, %6, %6) {lb = [-1, -1, -1], ub = [1, 1, 1]} : (!stencil.temp<68x66x62xf64>, index, index, index) -> f64
    #     %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #   }) {lb = [-1, 0, 0], ub = [65, 64, 60]} : (!stencil.temp<68x66x62xf64>) -> !stencil.temp<66x64x60xf64>
    #   %5 = "stencil.apply"(%4) ( {
    #   ^bb0(%arg2: !stencil.temp<66x64x60xf64>):  // no predecessors
    #     %6 = "stencil.access"(%arg2) {offset = [-1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %7 = "stencil.access"(%arg2) {offset = [1, 0, 0]} : (!stencil.temp<66x64x60xf64>) -> f64
    #     %8 = "std.addf"(%7, %6) : (f64, f64) -> f64
    #     %9 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
    #     "stencil.return"(%9) : (!stencil.result<f64>) -> ()
    #   }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #   "stencil.store"(%5, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #   "std.return"() : () -> ()
    # }) {stencil.program, sym_name = "dyn_access", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()

    # // result:

    # // module  {
    # //   func @dyn_access(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.load %0([-2, -1, -2] : [66, 65, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<68x66x62xf64>
    # //     %3 = stencil.apply (%arg2 = %2 : !stencil.temp<68x66x62xf64>) -> !stencil.temp<68x66x62xf64> {
    # //       %5 = stencil.access %arg2 [0, 0, -1] : (!stencil.temp<68x66x62xf64>) -> f64
    # //       %6 = stencil.store_result %5 : (f64) -> !stencil.result<f64>
    # //       stencil.return %6 : !stencil.result<f64>
    # //     } to ([-2, -1, -1] : [66, 65, 61])
    # //     %4 = stencil.apply (%arg2 = %3 : !stencil.temp<68x66x62xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %5 = stencil.index 0 [-1, 0, 0] : index
    # //       %6 = stencil.dyn_access %arg2(%5, %5, %5) in [-2, -1, -1] : [0, 1, 1] : (!stencil.temp<68x66x62xf64>) -> f64
    # //       %7 = stencil.index 0 [1, 0, 0] : index
    # //       %8 = stencil.dyn_access %arg2(%7, %7, %7) in [0, -1, -1] : [2, 1, 1] : (!stencil.temp<68x66x62xf64>) -> f64
    # //       %9 = addf %8, %6 : f64
    # //       %10 = stencil.store_result %9 : (f64) -> !stencil.result<f64>
    # //       stencil.return %10 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    parse(before)
    parse(after)

    apply_strategy_and_compare(
        before, after,
        seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_inlining_simple_buffer():
    unchanging = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  %3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %5 : !f64 = stencil.access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %6 : !stencil.result<!f64> = stencil.store_result(%5 : !f64)
    stencil.return(%6 : !stencil.result<!f64>)
  }
  %7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.buffer(%3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  %8 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%7 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^2(%9 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %10 : !f64 = stencil.access(%9 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %11 : !stencil.result<!f64> = stencil.store_result(%10 : !f64)
    stencil.return(%11 : !stencil.result<!f64>)
  }
  stencil.store(%8 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    # before:

    # func @simple_buffer(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    #   %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #   %2 = stencil.load %0([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
    #   %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    #     %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    #     %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    #     stencil.return %7 : !stencil.result<f64>
    #   } to ([0, 0, 0] : [64, 64, 60])
    #   %4 = stencil.buffer %3([0, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #   %5 = stencil.apply (%arg2 = %4 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    #     %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    #     %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    #     stencil.return %7 : !stencil.result<f64>
    #   } to ([0, 0, 0] : [64, 64, 60])
    #   stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    #   return
    # }

    # // result:

    # // module  {
    # //   func @simple_buffer(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.load %0([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
    # //     %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    # //       %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    # //       stencil.return %7 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     %4 = stencil.buffer %3([0, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    # //     %5 = stencil.apply (%arg2 = %4 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    # //       %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    # //       stencil.return %7 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }
    parse(unchanging)
    apply_strategy_and_compare(
        unchanging, unchanging,
        seq(try_(topToBottom(InlineProducer())),
            topToBottom(GarbageCollect())))


if __name__ == "__main__":
    # test_inlining_simple()
    # test_inlining_simple_index()
    # test_inlining_multiple_edges()
    # test_inlining_simple_ifelse()
    # test_inlining_reroute()
    # test_inlining_avoid_redundant()
    test_inlining_root()
    # test_inlining_dyn_access()
    # test_inlining_simple_buffer()