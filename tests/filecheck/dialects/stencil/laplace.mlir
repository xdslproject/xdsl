// RUN: xdsl-opt %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i32, -1 : i32], f64>, %1 : !stencil.field<[-1 : i32, -1 : i32], f64>):
    %2 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32], f64>
    %3 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32], f64>
    %4 = "stencil.load"(%2) : (!stencil.field<[72 : i32, 72 : i32], f64>) -> !stencil.temp<[66 : i32, 66 : i32], f64>
    %5 = "stencil.apply"(%4) ({
    ^1(%6 : !stencil.temp<[66 : i32, 66 : i32], f64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<[-1 : i32, 0 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<[1 : i32, 0 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, 1 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, -1 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, 0 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
      %12 = "arith.addf"(%7, %8) : (f64, f64) -> f64
      %13 = "arith.addf"(%9, %10) : (f64, f64) -> f64
      %14 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %16 = "arith.mulf"(%11, %15) : (f64, f64) -> f64
      %17 = "arith.mulf"(%16, %13) : (f64, f64) -> f64
      "stencil.return"(%17) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> !stencil.temp<[64 : i32, 64 : i32], f64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<[0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32], f64>, !stencil.field<[72 : i32, 72 : i32], f64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "stencil_laplace", "function_type" = (!stencil.field<[-1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : !stencil.field<[-1 : i32, -1 : i32], f64>, %1 : !stencil.field<[-1 : i32, -1 : i32], f64>):
// CHECK-NEXT:     %2 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32], f64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i32, -4 : i32]>, "ub" = #stencil.index<[68 : i32, 68 : i32]>} : (!stencil.field<[-1 : i32, -1 : i32], f64>) -> !stencil.field<[72 : i32, 72 : i32], f64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[72 : i32, 72 : i32], f64>) -> !stencil.temp<[66 : i32, 66 : i32], f64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^1(%6 : !stencil.temp<[66 : i32, 66 : i32], f64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<[-1 : i32, 0 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
// CHECK-NEXT:       %8 = "stencil.access"(%6) {"offset" = #stencil.index<[1 : i32, 0 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
// CHECK-NEXT:       %9 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, 1 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
// CHECK-NEXT:       %10 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, -1 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
// CHECK-NEXT:       %11 = "stencil.access"(%6) {"offset" = #stencil.index<[0 : i32, 0 : i32]>} : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> f64
// CHECK-NEXT:       %12 = "arith.addf"(%7, %8) : (f64, f64) -> f64
// CHECK-NEXT:       %13 = "arith.addf"(%9, %10) : (f64, f64) -> f64
// CHECK-NEXT:       %14 = "arith.addf"(%12, %13) : (f64, f64) -> f64
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %16 = "arith.mulf"(%11, %15) : (f64, f64) -> f64
// CHECK-NEXT:       %17 = "arith.mulf"(%16, %13) : (f64, f64) -> f64
// CHECK-NEXT:       "stencil.return"(%17) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[66 : i32, 66 : i32], f64>) -> !stencil.temp<[64 : i32, 64 : i32], f64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<[0 : i32, 0 : i32]>, "ub" = #stencil.index<[64 : i32, 64 : i32]>} : (!stencil.temp<[64 : i32, 64 : i32], f64>, !stencil.field<[72 : i32, 72 : i32], f64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "stencil_laplace", "function_type" = (!stencil.field<[-1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
