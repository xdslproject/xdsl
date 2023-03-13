// RUN: xdsl-opt %s -t mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, %1 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
    %2 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i32, -4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>
    %3 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i32, -4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>
    %4 = "stencil.load"(%2) {"lb" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32, 64 : i32]>} : (!stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>) -> !stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>
    %5 = "stencil.apply"(%4) ({
    ^1(%6 : !stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>) -> f64
      %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
      "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32, 64 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>) -> !stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32, 64 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>, !stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "stencil_copy", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, %1 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
// CHECK-NEXT:     %2 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i32, -4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i32, -4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) {"lb" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32, 64 : i32]>} : (!stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>) -> !stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^1(%6 : !stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>) -> f64
// CHECK-NEXT:       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:     }) {"lb" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32, 64 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>) -> !stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<[0 : i32, 0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32, 64 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32, 64 : i32], f64>, !stencil.field<[72 : i32, 72 : i32, 72 : i32], f64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "stencil_copy", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
