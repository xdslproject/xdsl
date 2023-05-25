// RUN: xdsl-opt %s --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
    %2 = "stencil.cast"(%0) {"lb" = #stencil.index<-4, -4, -4>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) {"lb" = #stencil.index<-4, -4, -4>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    %5 = "stencil.apply"(%4) ({
    ^1(%6 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> f64
      %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
      "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "stencil_copy", "function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
// CHECK-NEXT:     %2 = "stencil.cast"(%0) {"lb" = #stencil.index<-4, -4, -4>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) {"lb" = #stencil.index<-4, -4, -4>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^1(%6 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "stencil_copy", "function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
