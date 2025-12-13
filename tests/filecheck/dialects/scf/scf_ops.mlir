// RUN: XDSL_ROUNDTRIP


builtin.module {


  %0 = "test.op"() : () -> i1
  scf.if %0 {
    %1 = "test.op"() : () -> i32
    scf.yield
  } else {
    %2 = "test.op"() : () -> i32
  }

  // CHECK:      %{{.*}} = "test.op"() : () -> i1
  // CHECK-NEXT: scf.if %{{.*}} {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT: }


  %3 = scf.if %0 -> (i32) {
    %4 = "test.op"() : () -> i32
    scf.yield %4 : i32
  } else {
    %5 = "test.op"() : () -> i32
    scf.yield %5 : i32
  }

  // CHECK:      %{{.*}} = scf.if %{{.*}} -> (i32) {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   scf.yield %{{.*}} : i32
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   scf.yield %{{.*}} : i32
  // CHECK-NEXT: }

  scf.if %0 {}

  // CHECK: scf.if %{{.*}} {
  // CHECK-NEXT: }

  func.func @while() {
    %init = arith.constant 0 : i32
    %res = scf.while (%arg = %init) : (i32) -> i32 {
      %zero = arith.constant 0 : i32
      %c = "arith.cmpi"(%zero, %arg) {"predicate" = 1 : i64} : (i32, i32) -> i1
      scf.condition(%c) %zero : i32
    } do {
    ^bb1(%arg2 : i32):
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
  // CHECK-NEXT:   ^bb{{.*}}(%{{.*}} : i32):
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
  // CHECK-NEXT:    ^bb{{\d+}}(%{{.*}} : i32, %{{.*}} : f32):
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
  // CHECK-NEXT:      scf.condition(%{{.*}}) {hello = "world"} %{{.*}}, %{{.*}} : i32, f32
  // CHECK-NEXT:    } do {
  // CHECK-NEXT:    ^bb{{.*}}(%{{.*}} : i32, %{{.*}} : f32):
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
    %res_1 = scf.for %iv = %lb to %ub step %s iter_args(%prod_iter = %prod) -> (index) {
      %prod_new = arith.muli %prod_iter, %iv : index
      scf.yield %prod_new : index
    }
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

  func.func @for_i32() {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 42 : i32
    %s = arith.constant 3 : i32
    %prod = arith.constant 1 : i32
    %res_1 = scf.for %iv = %lb to %ub step %s iter_args(%prod_iter = %prod) -> (i32) : i32 {
      %prod_new = arith.muli %prod_iter, %iv : i32
      scf.yield %prod_new : i32
    }
    func.return
  }

  // CHECK-NEXT: func.func @for_i32() {
  // CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
  // CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32
  // CHECK-NEXT:   %{{.*}} = arith.constant 3 : i32
  // CHECK-NEXT:   %{{.*}} = arith.constant 1 : i32
  // CHECK-NEXT:   %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) : i32 {
  // CHECK-NEXT:     %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:     scf.yield %{{.*}} : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @index_switch(%flag: index) -> i32 {
    %a = arith.constant 0 : i32
    %b = arith.constant 1 : i32
    %c, %d = scf.index_switch %flag -> i32, i32
    case 1 {
      scf.yield %a, %a : i32, i32
    }
    default {
      scf.yield %b, %b : i32, i32
    }
    func.return %c : i32
  }

  // CHECK:      func.func @index_switch(%flag : index) -> i32 {
  // CHECK-NEXT:   %a = arith.constant 0 : i32
  // CHECK-NEXT:   %b = arith.constant 1 : i32
  // CHECK-NEXT:   %c, %d = scf.index_switch %flag -> i32, i32
  // CHECK-NEXT:   case 1 {
  // CHECK-NEXT:     scf.yield %a, %a : i32, i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   default {
  // CHECK-NEXT:     scf.yield %b, %b : i32, i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return %c : i32
  // CHECK-NEXT: }

  func.func @switch_trivial(%flag: index) {
    scf.index_switch %flag
    default {
      scf.yield
    }
    func.return
  }

  // CHECK:      func.func @switch_trivial(%flag : index) {
  // CHECK-NEXT:   scf.index_switch %flag
  // CHECK-NEXT:   default {
  // CHECK-NEXT:     scf.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @execute_region() {
    %b = scf.execute_region -> (i32) {
      %a = "test.op"() : () -> (i32)
      scf.yield %a : i32
    }
    func.return
  }
  
  // CHECK:      func.func @execute_region() {
  // CHECK-NEXT:   %b = scf.execute_region -> (i32) {
  // CHECK-NEXT:     %a = "test.op"() : () -> i32
  // CHECK-NEXT:     scf.yield %a : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

    func.func @execute_region_multiple_results() {
        %c, %d = scf.execute_region -> (i32, i64) {
        %a, %b = "test.op"() : () -> (i32, i64)
        scf.yield %a, %b : i32, i64
        }
        func.return
    }
    
    // CHECK:      func.func @execute_region_multiple_results() {
    // CHECK-NEXT:   %c, %d = scf.execute_region -> (i32, i64) {
    // CHECK-NEXT:     %a, %b = "test.op"() : () -> (i32, i64)
    // CHECK-NEXT:     scf.yield %a, %b : i32, i64
    // CHECK-NEXT:   }
    // CHECK-NEXT:   func.return
    // CHECK-NEXT: }

  func.func @execute_region_multiple_blocks() {
    %c = scf.execute_region -> (i32) {
      %cond = "test.op"() : () -> i1
      cf.cond_br %cond, ^bb0, ^bb1
    ^bb0:
      %a = "test.op"() : () -> i32
      scf.yield %a : i32
    ^bb1:
      %b = "test.op"() : () -> i32
      scf.yield %b : i32
    }
    func.return
  }

  // CHECK:      func.func @execute_region_multiple_blocks() {
  // CHECK-NEXT:   %c = scf.execute_region -> (i32) {
  // CHECK-NEXT:     %cond = "test.op"() : () -> i1
  // CHECK-NEXT:     cf.cond_br %cond, ^bb0, ^bb1
  // CHECK-NEXT:   ^bb0:
  // CHECK-NEXT:     %a = "test.op"() : () -> i32
  // CHECK-NEXT:     scf.yield %a : i32
  // CHECK-NEXT:   ^bb1:
  // CHECK-NEXT:     %b = "test.op"() : () -> i32
  // CHECK-NEXT:     scf.yield %b : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

}
