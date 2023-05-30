// RUN: xdsl-opt %s --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>):
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]xf64>
    %5 = "stencil.apply"(%4) ({
    ^1(%6 : !stencil.temp<[-1,65]x[-1,65]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %12 = "arith.addf"(%7, %8) : (f64, f64) -> f64
      %13 = "arith.addf"(%9, %10) : (f64, f64) -> f64
      %14 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %16 = "arith.mulf"(%11, %15) : (f64, f64) -> f64
      %17 = "arith.mulf"(%16, %13) : (f64, f64) -> f64
      "stencil.return"(%17) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> !stencil.temp<[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<64, 64>} : (!stencil.temp<[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "stencil_laplace", "function_type" = (!stencil.field<?x?xf64>, !stencil.field<?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>):
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^1(%6 : !stencil.temp<[-1,65]x[-1,65]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %12 = "arith.addf"(%7, %8) : (f64, f64) -> f64
// CHECK-NEXT:       %13 = "arith.addf"(%9, %10) : (f64, f64) -> f64
// CHECK-NEXT:       %14 = "arith.addf"(%12, %13) : (f64, f64) -> f64
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %16 = "arith.mulf"(%11, %15) : (f64, f64) -> f64
// CHECK-NEXT:       %17 = "arith.mulf"(%16, %13) : (f64, f64) -> f64
// CHECK-NEXT:       "stencil.return"(%17) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> !stencil.temp<[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<64, 64>} : (!stencil.temp<[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "stencil_laplace", "function_type" = (!stencil.field<?x?xf64>, !stencil.field<?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
