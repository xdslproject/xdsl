// RUN: XDSL_ROUNDTRIP


builtin.module {


  %0 = "test.op"() : () -> i1
  "scf.if"(%0) ({
    %1 = "test.op"() : () -> i32
    scf.yield
  }, {
    %2 = "test.op"() : () -> i32
    scf.yield
  }) : (i1) -> ()

  // CHECK:      %{{.*}} = "test.op"() : () -> i1
  // CHECK-NEXT: "scf.if"(%{{.*}}) ({
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   scf.yield
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   scf.yield
  // CHECK-NEXT: }) : (i1) -> ()


  %3 = "scf.if"(%0) ({
    %4 = "test.op"() : () -> i32
    scf.yield %4 : i32
  }, {
    %5 = "test.op"() : () -> i32
    scf.yield %5 : i32
  }) : (i1) -> i32


  // CHECK:      %{{.*}} = "scf.if"(%{{.*}}) ({
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   scf.yield %{{.*}} : i32
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   scf.yield %{{.*}} : i32
  // CHECK-NEXT: }) : (i1) -> i32

  func.func @while() {
    %init = arith.constant 0 : i32
    %res = scf.while (%arg = %init) : (i32) -> i32 {
      %zero = arith.constant 0 : i32
      %c = "arith.cmpi"(%zero, %arg) {"predicate" = 1 : i64} : (i32, i32) -> i1
      scf.condition(%c) %zero : i32
    } do {
    ^1(%arg2 : i32):
      scf.yield %arg2 : i32
    }
    func.return
  }

  // CHECK:      func.func @while() {
  // CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
  // CHECK-NEXT:   %{{.*}} = scf.while (%{{.*}} = %{{.*}}) : (i32) -> i32 {
  // CHECK-NEXT:     %{{.*}} = arith.constant 0 : i32
  // CHECK-NEXT:     %{{.*}} = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:     scf.condition(%{{.*}}) %{{.*}} : i32
  // CHECK-NEXT:   } do {
  // CHECK-NEXT:   ^{{.*}}(%{{.*}} : i32):
  // CHECK-NEXT:     scf.yield %{{.*}} : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }


  func.func @while2() {
    %a = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %b = "arith.constant"() {value = 32 : i32} : () -> i32
    %2:2 = scf.while (%arg0 = %b, %arg1 = %a) : (i32, f32) -> (i32, f32) {
      %c = "arith.constant"() {value = 0 : i32} : () -> i32
      %d = "arith.cmpi"(%arg0, %c) {predicate = 0 : i64} : (i32, i32) -> i1
      scf.condition(%d) %arg0, %arg1 : i32, f32
    } do {
    ^bb0(%arg0: i32, %arg1: f32):
      %c = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
      %d = "arith.addf"(%c, %arg1) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      scf.yield %arg0, %d : i32, f32
    }
    func.return
  }

  // CHECK-NEXT:  func.func @while2() {
  // CHECK-NEXT:    %{{.*}} = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:    %{{.*}} = arith.constant 32 : i32
  // CHECK-NEXT:    %6, %7 = scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, f32) -> (i32, f32) {
  // CHECK-NEXT:      %{{.*}} = arith.constant 0 : i32
  // CHECK-NEXT:      %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:      scf.condition(%{{.*}}) %{{.*}}, %{{.*}} : i32, f32
  // CHECK-NEXT:    } do {
  // CHECK-NEXT:    ^{{\d+}}(%{{.*}} : i32, %{{.*}} : f32):
  // CHECK-NEXT:      %{{.*}} = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      scf.yield %{{.*}}, %{{.*}} : i32, f32
  // CHECK-NEXT:    }
  // CHECK-NEXT:    func.return
  // CHECK-NEXT:  }


  func.func @while3() {
    %a = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %b = "arith.constant"() {value = 32 : i32} : () -> i32
    %2:2 = scf.while (%arg0 = %b, %arg1 = %a) : (i32, f32) -> (i32, f32) {
      %c = "arith.constant"() {value = 0 : i32} : () -> i32
      %d = "arith.cmpi"(%arg0, %c) {predicate = 0 : i64} : (i32, i32) -> i1
      scf.condition(%d) {"hello" = "world"} %arg0, %arg1 : i32, f32
    } do {
    ^bb0(%arg0: i32, %arg1: f32):
      %c = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
      %d = "arith.addf"(%c, %arg1) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      scf.yield %arg0, %d : i32, f32
    }
    func.return
  }

  // CHECK-NEXT:  func.func @while3() {
  // CHECK-NEXT:    %{{.*}} = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:    %{{.*}} = arith.constant 32 : i32
  // CHECK-NEXT:    %{{.*}}, %{{.*}} = scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, f32) -> (i32, f32) {
  // CHECK-NEXT:      %{{.*}} = arith.constant 0 : i32
  // CHECK-NEXT:      %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:      scf.condition(%{{.*}}) {"hello" = "world"} %{{.*}}, %{{.*}} : i32, f32
  // CHECK-NEXT:    } do {
  // CHECK-NEXT:    ^{{.*}}(%{{.*}} : i32, %{{.*}} : f32):
  // CHECK-NEXT:      %{{.*}} = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      scf.yield %{{.*}}, %{{.*}} : i32, f32
  // CHECK-NEXT:    }
  // CHECK-NEXT:    func.return
  // CHECK-NEXT:  }

  func.func @for() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 42 : index
    %s = arith.constant 3 : index
    %prod = arith.constant 1 : index
    %res_1 = "scf.for"(%lb, %ub, %s, %prod) ({
    ^2(%iv : index, %prod_iter : index):
      %prod_new = arith.muli %prod_iter, %iv : index
      scf.yield %prod_new : index
    }) : (index, index, index, index) -> index
    func.return
  }

  // CHECK-NEXT: func.func @for() {
  // CHECK-NEXT:   %{{.*}} = arith.constant 0 : index
  // CHECK-NEXT:   %{{.*}} = arith.constant 42 : index
  // CHECK-NEXT:   %{{.*}} = arith.constant 3 : index
  // CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
  // CHECK-NEXT:   %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
  // CHECK-NEXT:     %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
  // CHECK-NEXT:     scf.yield %{{.*}} : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }


}
