// RUN: XDSL_AUTO_ROUNDTRIP


builtin.module {

  %0 = "test.op"() : () -> i1
  "scf.if"(%0) ({
    %1 = "test.op"() : () -> i32
    "scf.yield"() : () -> ()
  }, {
    %2 = "test.op"() : () -> i32
    "scf.yield"() : () -> ()
  }) : (i1) -> ()


  %3 = "scf.if"(%0) ({
    %4 = "test.op"() : () -> i32
    "scf.yield"(%4) : (i32) -> ()
  }, {
    %5 = "test.op"() : () -> i32
    "scf.yield"(%5) : (i32) -> ()
  }) : (i1) -> i32

  func.func @while() {
    %init = arith.constant 0 : i32
    %res = "scf.while"(%init) ({
    ^0(%arg : i32):
      %zero = arith.constant 0 : i32
      %c = "arith.cmpi"(%zero, %arg) {"predicate" = 1 : i64} : (i32, i32) -> i1
      "scf.condition"(%c, %zero) : (i1, i32) -> ()
    }, {
    ^1(%arg2 : i32):
      "scf.yield"(%arg2) : (i32) -> ()
    }) : (i32) -> i32
    func.return
  }

  func.func @while2() {
    %a = arith.constant 1.000000e+00 : f32
    %b = arith.constant 32 : i32
    %6, %7 = "scf.while"(%b, %a) ({
    ^2(%arg0 : i32, %arg1 : f32):
      %c_1 = arith.constant 0 : i32
      %d = "arith.cmpi"(%arg0, %c_1) {"predicate" = 0 : i64} : (i32, i32) -> i1
      "scf.condition"(%d, %arg0, %arg1) : (i1, i32, f32) -> ()
    }, {
    ^3(%arg0_1 : i32, %arg1_1 : f32):
      %c_2 = arith.constant 1.000000e+00 : f32
      %d_1 = arith.addf %c_2, %arg1_1 : f32
      "scf.yield"(%arg0_1, %d_1) : (i32, f32) -> ()
    }) : (i32, f32) -> (i32, f32)
    func.return
  }

  func.func @for() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 42 : index
    %s = arith.constant 3 : index
    %prod = arith.constant 1 : index
    %res_1 = "scf.for"(%lb, %ub, %s, %prod) ({
    ^4(%iv : index, %prod_iter : index):
      %prod_new = arith.muli %prod_iter, %iv : index
      "scf.yield"(%prod_new) : (index) -> ()
    }) : (index, index, index, index) -> index
    func.return
  }
}
