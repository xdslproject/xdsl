// RUN: xdsl-opt %s -t mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%arg0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %0 = "stencil.cast"(%arg0) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %1 = "stencil.cast"(%arg1) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %2 = "stencil.load"(%0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[66 : i64, 66 : i64, 63 : i64]>} : (!stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>
    %3 = "stencil.apply"(%2) ({
    ^1(%arg2 : !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>):
      %4 = "stencil.access"(%arg2) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %5 = "stencil.access"(%arg2) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %6 = "arith.addf"(%4, %5) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
      %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
      "stencil.return"(%7) : (!stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>, "ub" = #stencil.index<[65 : i64, 66 : i64, 63 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>
    %8, %9 = "stencil.apply"(%2, %3) ({
    ^2(%10 : !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, %arg3 : !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>):
      %11 = "stencil.access"(%10) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %12 = "stencil.access"(%arg3) {"offset" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>) -> f64
      %13 = "arith.subf"(%11, %12) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
      %14 = "stencil.store_result"(%13) : (f64) -> !stencil.result<f64>
      "stencil.return"(%14, %14) : (!stencil.result<f64>, !stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>) -> (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>)
    "stencil.store"(%8, %1) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
    "stencil.store"(%9, %0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "simple_stencil_inlining"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%arg0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
// CHECK-NEXT:     %0 = "stencil.cast"(%arg0) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
// CHECK-NEXT:     %1 = "stencil.cast"(%arg1) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
// CHECK-NEXT:     %2 = "stencil.load"(%0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[66 : i64, 66 : i64, 63 : i64]>} : (!stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>
// CHECK-NEXT:     %3 = "stencil.apply"(%2) ({
// CHECK-NEXT:     ^1(%arg2 : !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>):
// CHECK-NEXT:       %4 = "stencil.access"(%arg2) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
// CHECK-NEXT:       %5 = "stencil.access"(%arg2) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
// CHECK-NEXT:       %6 = "arith.addf"(%4, %5) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
// CHECK-NEXT:       %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%7) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:     }) {"lb" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>, "ub" = #stencil.index<[65 : i64, 66 : i64, 63 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>
// CHECK-NEXT:     %8, %9 = "stencil.apply"(%2, %3) ({
// CHECK-NEXT:     ^2(%10 : !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, %arg3 : !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>):
// CHECK-NEXT:       %11 = "stencil.access"(%10) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
// CHECK-NEXT:       %12 = "stencil.access"(%arg3) {"offset" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>) -> f64
// CHECK-NEXT:       %13 = "arith.subf"(%11, %12) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
// CHECK-NEXT:       %14 = "stencil.store_result"(%13) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%14, %14) : (!stencil.result<f64>, !stencil.result<f64>) -> ()
// CHECK-NEXT:     }) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>) -> (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>)
// CHECK-NEXT:     "stencil.store"(%8, %1) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
// CHECK-NEXT:     "stencil.store"(%9, %0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "simple_stencil_inlining"} : () -> ()
// CHECK-NEXT: }) : () -> ()
