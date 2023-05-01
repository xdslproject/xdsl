// RUN: xdsl-opt %s | xdsl-opt | filecheck %s


"builtin.module"() ({


  %0 = "test.op"() : () -> i1
  "scf.if"(%0) ({
    %1 = "test.op"() : () -> i32
    "scf.yield"() : () -> ()
  }, {
    %2 = "test.op"() : () -> i32
    "scf.yield"() : () -> ()
  }) : (i1) -> ()

  // CHECK:      %{{.*}} = "test.op"() : () -> i1
  // CHECK-NEXT: "scf.if"(%{{.*}}) ({
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   "scf.yield"() : () -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   "scf.yield"() : () -> ()
  // CHECK-NEXT: }) : (i1) -> ()


  %3 = "scf.if"(%0) ({
    %4 = "test.op"() : () -> i32
    "scf.yield"(%4) : (i32) -> ()
  }, {
    %5 = "test.op"() : () -> i32
    "scf.yield"(%5) : (i32) -> ()
  }) : (i1) -> i32


  // CHECK:      %{{.*}} = "scf.if"(%{{.*}}) ({
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   "scf.yield"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
  // CHECK-NEXT:   "scf.yield"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT: }) : (i1) -> i32

  "func.func"() ({
    %init = "arith.constant"() {"value" = 0 : i32} : () -> i32
    %res = "scf.while"(%init) ({
    ^0(%arg : i32):
      %zero = "arith.constant"() {"value" = 0 : i32} : () -> i32
      %c = "arith.cmpi"(%zero, %arg) {"predicate" = 1 : i64} : (i32, i32) -> i1
      "scf.condition"(%c, %zero) : (i1, i32) -> ()
    }, {
    ^1(%arg2 : i32):
      "scf.yield"(%arg2) : (i32) -> ()
    }) : (i32) -> i32
    "func.return"() : () -> ()
  }) {"sym_name" = "while", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  // CHECK:      "func.func"() ({
  // CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 0 : i32} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "scf.while"(%{{.*}}) ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}} : i32):
  // CHECK-NEXT:     %{{.*}} = "arith.constant"() {"value" = 0 : i32} : () -> i32
  // CHECK-NEXT:     %{{.*}} = "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 1 : i64} : (i32, i32) -> i1
  // CHECK-NEXT:     "scf.condition"(%{{.*}}, %{{.*}}) : (i1, i32) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   ^{{.*}}(%{{.*}} : i32):
  // CHECK-NEXT:     "scf.yield"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT:   }) : (i32) -> i32
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }) {"sym_name" = "while", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
    %lb = "arith.constant"() {"value" = 0 : index} : () -> index
    %ub = "arith.constant"() {"value" = 42 : index} : () -> index
    %s = "arith.constant"() {"value" = 3 : index} : () -> index
    %prod = "arith.constant"() {"value" = 1 : index} : () -> index
    %res_1 = "scf.for"(%lb, %ub, %s, %prod) ({
    ^2(%iv : index, %prod_iter : index):
      %prod_new = "arith.muli"(%prod_iter, %iv) : (index, index) -> index
      "scf.yield"(%prod_new) : (index) -> ()
    }) : (index, index, index, index) -> index
    "func.return"() : () -> ()
  }) {"sym_name" = "for", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  // CHECK-NEXT: "func.func"() ({
  // CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 0 : index} : () -> index
  // CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 42 : index} : () -> index
  // CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 3 : index} : () -> index
  // CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 1 : index} : () -> index
  // CHECK-NEXT:   %{{.*}} = "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
  // CHECK-NEXT:     %{{.*}} = "arith.muli"(%{{.*}}, %{{.*}}) : (index, index) -> index
  // CHECK-NEXT:     "scf.yield"(%{{.*}}) : (index) -> ()
  // CHECK-NEXT:   }) : (index, index, index, index) -> index
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }) {"sym_name" = "for", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()


}) : () -> ()
