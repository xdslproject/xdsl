// RUN: xdsl-opt %s -t mlir -f mlir | xdsl-opt -t mlir -f mlir | FileCheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0:
    "cf.br"() [^1] : () -> ()
  ^1:
    "cf.br"() [^1] : () -> ()
  }) {"sym_name" = "unconditional_br", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
  // CHECK: "func.func"() ({
  // CHECK-NEXT:  ^{{.*}}:
  // CHECK-NEXT:    "cf.br"() [^{{.*}}] : () -> ()
  // CHECK-NEXT:  ^{{.*}}:
  // CHECK-NEXT:    "cf.br"() [^{{.*}}] : () -> ()
  // CHECK-NEXT:}) {"sym_name" = "unconditional_br", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  ^2(%0 : i32):
    "cf.br"(%0) [^3] : (i32) -> ()
  ^3(%1 : i32):
    "cf.br"(%1) [^3] : (i32) -> ()
  }) {"sym_name" = "br", "function_type" = (i32) -> (), "sym_visibility" = "private"} : () -> ()
  // CHECK: "func.func"() ({
  // CHECK-NEXT:  ^{{.*}}(%{{.*}} : i32):
  // CHECK-NEXT:    "cf.br"(%{{.*}}) [^{{.*}}] : (i32) -> ()
  // CHECK-NEXT:  ^{{.*}}(%{{.*}} : i32):
  // CHECK-NEXT:    "cf.br"(%{{.*}}) [^{{.*}}] : (i32) -> ()
  // CHECK-NEXT:}) {"sym_name" = "br", "function_type" = (i32) -> (), "sym_visibility" = "private"} : () -> ()
  

  "func.func"() ({
  ^4(%2 : i1, %3 : i32):
    "cf.br"(%2, %3) [^5] : (i1, i32) -> ()
  ^5(%4 : i1, %5 : i32):
    "cf.cond_br"(%4, %4, %5, %5, %5, %5) [^5, ^6] {"operand_segment_sizes" = array<i32: 1, 2, 3>} : (i1, i1, i32, i32, i32, i32) -> ()
  ^6(%6 : i32, %7 : i32, %8 : i32):
    "func.return"(%6) : (i32) -> ()
  }) {"sym_name" = "cond_br", "function_type" = (i1, i32) -> i32, "sym_visibility" = "private"} : () -> ()
  // CHECK: "func.func"() ({
  // CHECK-NEXT:  ^{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-NEXT:    "cf.br"(%{{.*}}, %{{.*}}) [^{{.*}}] : (i1, i32) -> ()
  // CHECK-NEXT:  ^{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-NEXT:    "cf.cond_br"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) [^{{.*}}, ^{{.*}}] {"operand_segment_sizes" = array<i32: 1, 2, 3>} : (i1, i1, i32, i32, i32, i32) -> ()
  // CHECK-NEXT:  ^{{.*}}(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
  // CHECK-NEXT:    "func.return"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT:}) {"sym_name" = "cond_br", "function_type" = (i1, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
