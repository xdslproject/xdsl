// RUN: xdsl-opt %s -t mlir -p stencil-inlining | filecheck %s

"builtin.module"() ( {
  "func.func"() ( {
  ^bb0(%arg0: !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg1: !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg2: f64):
    %0 = "stencil.cast"(%arg0) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %1 = "stencil.cast"(%arg1) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %2 = "stencil.load"(%0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[66 : i64, 66 : i64, 63 : i64]>} : (!stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>
    %3 = "stencil.apply"(%2, %arg2) ( {
    ^bb0(%arg3: !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, %arg4: f64):
      %5 = "stencil.access"(%arg3) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %6 = "stencil.access"(%arg3) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %7 = "arith.addf"(%5, %6) : (f64, f64) -> f64
      %8 = "arith.addf"(%7, %arg4) : (f64, f64) -> f64
      %9 = "stencil.store_result"(%8) : (f64) -> !stencil.result<f64>
      "stencil.return"(%9) : (!stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>, "ub" = #stencil.index<[65 : i64, 66 : i64, 63 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, f64) -> !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>
    %4 = "stencil.apply"(%arg2, %3) ( {
    ^bb0(%arg3: f64, %arg4: !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>):
      %5 = "stencil.access"(%arg4) {"offset" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>) -> f64
      %6 = "arith.subf"(%arg3, %5) : (f64, f64) -> f64
      %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
      "stencil.return"(%7) : (!stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (f64, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>) -> !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>
    "stencil.store"(%4, %1) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, f64) -> (), "sym_name" = "simple_stencil_inlining"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%arg0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg2 : f64):
// CHECK-NEXT:     %0 = "stencil.cast"(%arg0) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
// CHECK-NEXT:     %1 = "stencil.cast"(%arg1) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
// CHECK-NEXT:     %2 = "stencil.load"(%0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[66 : i64, 66 : i64, 63 : i64]>} : (!stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>
// CHECK-NEXT:     %3 = "stencil.apply"(%2, %arg2) ({
// CHECK-NEXT:     ^1(%4 : !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, %5 : f64):
// CHECK-NEXT:       %6 = "stencil.access"(%4) {"offset" = #stencil.index<[0 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
// CHECK-NEXT:       %7 = "stencil.access"(%4) {"offset" = #stencil.index<[2 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
// CHECK-NEXT:       %8 = "arith.addf"(%6, %7) : (f64, f64) -> f64
// CHECK-NEXT:       %9 = "arith.addf"(%8, %5) : (f64, f64) -> f64
// CHECK-NEXT:       %10 = "arith.subf"(%5, %9) : (f64, f64) -> f64
// CHECK-NEXT:       %11 = "stencil.store_result"(%10) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%11) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:     }) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, f64) -> !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>
// CHECK-NEXT:     "stencil.store"(%3, %1) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, f64) -> (), "sym_name" = "simple_stencil_inlining"} : () -> ()
// CHECK-NEXT: }) : () -> ()
