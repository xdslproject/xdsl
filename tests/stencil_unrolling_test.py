from __future__ import annotations
from io import StringIO
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil.stencil as stencil
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
from xdsl.dialects.stencil.stencil_unrolling import UnrollApplyOp
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


# All tests sourced from: https://github.com/spcl/open-earth-compiler/blob/master/test/Dialect/Stencil/stencil-unrolling.mlir


def test_unrolling_access():

    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70,70,70]>, %arg1 : !stencil.field<[70,70,70]>):
    %1 : !stencil.temp<[64,64,60]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    %2 : !stencil.temp<[64,64,60]>  = stencil.apply(%1 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^bb0(%arg2: !stencil.temp<[64,64,60]>): 
        %3 : !f64 = stencil.access(%arg2: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %4 : !stencil.result<!f64> = stencil.store_result(%3: !f64)
        stencil.return(%4: !stencil.result<!f64>)
    }
    stencil.store(%2 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  %3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %5 : !f64 = stencil.access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %6 : !stencil.result<!f64> = stencil.store_result(%5 : !f64)
    %7 : !f64 = stencil.access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 1 : !i64, 0 : !i64]]
    %8 : !stencil.result<!f64> = stencil.store_result(%7 : !f64)
    %9 : !f64 = stencil.access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 2 : !i64, 0 : !i64]]
    %10 : !stencil.result<!f64> = stencil.store_result(%9 : !f64)
    %11 : !f64 = stencil.access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["offset" = [0 : !i64, 3 : !i64, 0 : !i64]]
    %12 : !stencil.result<!f64> = stencil.store_result(%11 : !f64)
    stencil.return(%6 : !stencil.result<!f64>, %8 : !stencil.result<!f64>, %10 : !stencil.result<!f64>, %12 : !stencil.result<!f64>) ["unroll" = [1 : !i64, 4 : !i64, 1 : !i64]]
  }
  stencil.store(%3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    # source before

    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg0) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %2 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %4 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%5) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%3, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x60xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "access", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()

    # source after

    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg0) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %2 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %4 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    #       %6 = "stencil.access"(%arg2) {offset = [0, 1, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
    #       %8 = "stencil.access"(%arg2) {offset = [0, 2, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %9 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
    #       %10 = "stencil.access"(%arg2) {offset = [0, 3, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %11 = "stencil.store_result"(%10) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%5, %7, %9, %11) {unroll = [1, 4, 1]} : (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%3, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x60xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "access", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()

    parse(before)
    parse(after)
    apply_strategy_and_compare(before, after, topToBottom(UnrollApplyOp()))


def test_unrolling_index():

    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !f64, %arg1 : !stencil.field<[70,70,70]>):
    %1 : !stencil.temp<[64,64,60]>  = stencil.apply(%arg0 : !f64) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^bb0(%arg2: !f64): 
        %2 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0, 0, 0]]
        %3 : !index = arith.constant() ["value" = 20 : !index]
        %4 : !f64 = arith.constant() ["value" = 0 : !i32]
        %5 : !i1 = arith.cmpi(%2 : !index, %3 : !index) ["predicate" = 2 : !i64]
        %6 : !f64 = arith.select(%5 : !i1, %arg2 : !f64, %4: !f64)
        %7: !stencil.result<!f64> = stencil.store_result(%6: !f64)
        stencil.return(%7: !stencil.result<!f64>)
    }
    stencil.store(%1 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !f64, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%0 : !f64) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%3 : !f64):
    %4 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %5 : !index = arith.constant() ["value" = 20 : !index]
    %6 : !f64 = arith.constant() ["value" = 0 : !i32]
    %7 : !i1 = arith.cmpi(%4 : !index, %5 : !index) ["predicate" = 2 : !i64]
    %8 : !f64 = arith.select(%7 : !i1, %3 : !f64, %6 : !f64)
    %9 : !stencil.result<!f64> = stencil.store_result(%8 : !f64)
    %10 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 1 : !i64, 0 : !i64]]
    %11 : !index = arith.constant() ["value" = 20 : !index]
    %12 : !f64 = arith.constant() ["value" = 0 : !i32]
    %13 : !i1 = arith.cmpi(%10 : !index, %11 : !index) ["predicate" = 2 : !i64]
    %14 : !f64 = arith.select(%13 : !i1, %3 : !f64, %12 : !f64)
    %15 : !stencil.result<!f64> = stencil.store_result(%14 : !f64)
    %16 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 2 : !i64, 0 : !i64]]
    %17 : !index = arith.constant() ["value" = 20 : !index]
    %18 : !f64 = arith.constant() ["value" = 0 : !i32]
    %19 : !i1 = arith.cmpi(%16 : !index, %17 : !index) ["predicate" = 2 : !i64]
    %20 : !f64 = arith.select(%19 : !i1, %3 : !f64, %18 : !f64)
    %21 : !stencil.result<!f64> = stencil.store_result(%20 : !f64)
    %22 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 3 : !i64, 0 : !i64]]
    %23 : !index = arith.constant() ["value" = 20 : !index]
    %24 : !f64 = arith.constant() ["value" = 0 : !i32]
    %25 : !i1 = arith.cmpi(%22 : !index, %23 : !index) ["predicate" = 2 : !i64]
    %26 : !f64 = arith.select(%25 : !i1, %3 : !f64, %24 : !f64)
    %27 : !stencil.result<!f64> = stencil.store_result(%26 : !f64)
    stencil.return(%9 : !stencil.result<!f64>, %15 : !stencil.result<!f64>, %21 : !stencil.result<!f64>, %27 : !stencil.result<!f64>) ["unroll" = [1 : !i64, 4 : !i64, 1 : !i64]]
  }
  stencil.store(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    # source before

    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg1) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %1 = "stencil.apply"(%arg0) ( {
    #     ^bb0(%arg2: f64):  // no predecessors
    #       %2 = "stencil.index"() {dim = 2 : i64, offset = [0, 0, 0]} : () -> index
    #       %c20 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %3 = "std.cmpi"(%2, %c20) {predicate = 2 : i64} : (index, index) -> i1
    #       %4 = "std.select"(%3, %arg2, %cst) : (i1, f64, f64) -> f64
    #       %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%5) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (f64) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%1, %0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x60xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "index", type = (f64, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()

    # source after

    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg1) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %1 = "stencil.apply"(%arg0) ( {
    #     ^bb0(%arg2: f64):  // no predecessors
    #       %2 = "stencil.index"() {dim = 2 : i64, offset = [0, 0, 0]} : () -> index
    #       %c20 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %3 = "std.cmpi"(%2, %c20) {predicate = 2 : i64} : (index, index) -> i1
    #       %4 = "std.select"(%3, %arg2, %cst) : (i1, f64, f64) -> f64
    #       %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    #       %6 = "stencil.index"() {dim = 2 : i64, offset = [0, 1, 0]} : () -> index
    #       %c20_0 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst_1 = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %7 = "std.cmpi"(%6, %c20_0) {predicate = 2 : i64} : (index, index) -> i1
    #       %8 = "std.select"(%7, %arg2, %cst_1) : (i1, f64, f64) -> f64
    #       %9 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
    #       %10 = "stencil.index"() {dim = 2 : i64, offset = [0, 2, 0]} : () -> index
    #       %c20_2 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst_3 = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %11 = "std.cmpi"(%10, %c20_2) {predicate = 2 : i64} : (index, index) -> i1
    #       %12 = "std.select"(%11, %arg2, %cst_3) : (i1, f64, f64) -> f64
    #       %13 = "stencil.store_result"(%12) : (f64) -> !stencil.result<f64>
    #       %14 = "stencil.index"() {dim = 2 : i64, offset = [0, 3, 0]} : () -> index
    #       %c20_4 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst_5 = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %15 = "std.cmpi"(%14, %c20_4) {predicate = 2 : i64} : (index, index) -> i1
    #       %16 = "std.select"(%15, %arg2, %cst_5) : (i1, f64, f64) -> f64
    #       %17 = "stencil.store_result"(%16) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%5, %9, %13, %17) {unroll = [1, 4, 1]} : (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (f64) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%1, %0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x60xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "index", type = (f64, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()
    parse(before)
    parse(after)
    apply_strategy_and_compare(before, after, topToBottom(UnrollApplyOp()))


def test_unrolling_dyn_access():

    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70,70,70]>, %arg1 : !stencil.field<[70,70,70]>):
    %0 : !stencil.temp<[64,64,60]> = stencil.load(%arg0 : !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    %1 : !stencil.temp<[64,64,60]>  = stencil.apply(%0 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^bb0(%arg2: !stencil.temp<[64,64,60]>):
        %2 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0, 0, 0]]
        %3 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0, 0, 0]]
        %4 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0, 0, 0]]
        %5 : !f64 = stencil.dyn_access(%arg2: !stencil.temp<[64,64,60]>, %2 : !index, %3 : !index, %4 : !index) ["lb" = [0, 0, 0], "ub" = [0, 0, 0]]
        %6: !stencil.result<!f64> = stencil.store_result(%5: !f64)
        stencil.return(%6: !stencil.result<!f64>)
    }
    stencil.store(%1 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  %3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>):
    %5 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %6 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %7 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %8 : !f64 = stencil.dyn_access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %5 : !index, %6 : !index, %7 : !index) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %9 : !stencil.result<!f64> = stencil.store_result(%8 : !f64)
    %10 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 1 : !i64, 0 : !i64]]
    %11 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 1 : !i64, 0 : !i64]]
    %12 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 1 : !i64, 0 : !i64]]
    %13 : !f64 = stencil.dyn_access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %10 : !index, %11 : !index, %12 : !index) ["lb" = [0 : !i64, 1 : !i64, 0 : !i64], "ub" = [0 : !i64, 1 : !i64, 0 : !i64]]
    %14 : !stencil.result<!f64> = stencil.store_result(%13 : !f64)
    %15 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 2 : !i64, 0 : !i64]]
    %16 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 2 : !i64, 0 : !i64]]
    %17 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 2 : !i64, 0 : !i64]]
    %18 : !f64 = stencil.dyn_access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %15 : !index, %16 : !index, %17 : !index) ["lb" = [0 : !i64, 2 : !i64, 0 : !i64], "ub" = [0 : !i64, 2 : !i64, 0 : !i64]]
    %19 : !stencil.result<!f64> = stencil.store_result(%18 : !f64)
    %20 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 3 : !i64, 0 : !i64]]
    %21 : !index = stencil.index() ["dim" = 1 : !i64, "offset" = [0 : !i64, 3 : !i64, 0 : !i64]]
    %22 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [0 : !i64, 3 : !i64, 0 : !i64]]
    %23 : !f64 = stencil.dyn_access(%4 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %20 : !index, %21 : !index, %22 : !index) ["lb" = [0 : !i64, 3 : !i64, 0 : !i64], "ub" = [0 : !i64, 3 : !i64, 0 : !i64]]
    %24 : !stencil.result<!f64> = stencil.store_result(%23 : !f64)
    stencil.return(%9 : !stencil.result<!f64>, %14 : !stencil.result<!f64>, %19 : !stencil.result<!f64>, %24 : !stencil.result<!f64>) ["unroll" = [1 : !i64, 4 : !i64, 1 : !i64]]
  }
  stencil.store(%3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""

    # source before

    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg0) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %2 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %4 = "stencil.index"() {dim = 0 : i64, offset = [0, 0, 0]} : () -> index
    #       %5 = "stencil.index"() {dim = 1 : i64, offset = [0, 0, 0]} : () -> index
    #       %6 = "stencil.index"() {dim = 2 : i64, offset = [0, 0, 0]} : () -> index
    #       %7 = "stencil.dyn_access"(%arg2, %4, %5, %6) {lb = [0, 0, 0], ub = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>, index, index, index) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%3, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x60xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "dyn_access", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()

    # source after

    # "module"() ( {
    #   "func"() ( {
    #   ^bb0(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg0) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, 0], ub = [67, 67, 60]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    #     %2 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %4 = "stencil.index"() {dim = 0 : i64, offset = [0, 0, 0]} : () -> index
    #       %5 = "stencil.index"() {dim = 1 : i64, offset = [0, 0, 0]} : () -> index
    #       %6 = "stencil.index"() {dim = 2 : i64, offset = [0, 0, 0]} : () -> index
    #       %7 = "stencil.dyn_access"(%arg2, %4, %5, %6) {lb = [0, 0, 0], ub = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>, index, index, index) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       %9 = "stencil.index"() {dim = 0 : i64, offset = [0, 1, 0]} : () -> index
    #       %10 = "stencil.index"() {dim = 1 : i64, offset = [0, 1, 0]} : () -> index
    #       %11 = "stencil.index"() {dim = 2 : i64, offset = [0, 1, 0]} : () -> index
    #       %12 = "stencil.dyn_access"(%arg2, %9, %10, %11) {lb = [0, 1, 0], ub = [0, 1, 0]} : (!stencil.temp<64x64x60xf64>, index, index, index) -> f64
    #       %13 = "stencil.store_result"(%12) : (f64) -> !stencil.result<f64>
    #       %14 = "stencil.index"() {dim = 0 : i64, offset = [0, 2, 0]} : () -> index
    #       %15 = "stencil.index"() {dim = 1 : i64, offset = [0, 2, 0]} : () -> index
    #       %16 = "stencil.index"() {dim = 2 : i64, offset = [0, 2, 0]} : () -> index
    #       %17 = "stencil.dyn_access"(%arg2, %14, %15, %16) {lb = [0, 2, 0], ub = [0, 2, 0]} : (!stencil.temp<64x64x60xf64>, index, index, index) -> f64
    #       %18 = "stencil.store_result"(%17) : (f64) -> !stencil.result<f64>
    #       %19 = "stencil.index"() {dim = 0 : i64, offset = [0, 3, 0]} : () -> index
    #       %20 = "stencil.index"() {dim = 1 : i64, offset = [0, 3, 0]} : () -> index
    #       %21 = "stencil.index"() {dim = 2 : i64, offset = [0, 3, 0]} : () -> index
    #       %22 = "stencil.dyn_access"(%arg2, %19, %20, %21) {lb = [0, 3, 0], ub = [0, 3, 0]} : (!stencil.temp<64x64x60xf64>, index, index, index) -> f64
    #       %23 = "stencil.store_result"(%22) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8, %13, %18, %23) {unroll = [1, 4, 1]} : (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%3, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x60xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "dyn_access", type = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    #   "module_terminator"() : () -> ()
    # }) : () -> ()

    parse(before)
    parse(after)
    apply_strategy_and_compare(before, after, topToBottom(UnrollApplyOp()))


if __name__ == "__main__":
    test_unrolling_access()
    test_unrolling_index()
    test_unrolling_dyn_access()