// RUN: xdsl-opt -p stencil-shape-inference --verify-diagnostics --split-input-file %s | filecheck %s

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      "stencil.return"(%16) : (f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}


// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^0(%6 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %12 = arith.addf %7, %8 : f64
// CHECK-NEXT:       %13 = arith.addf %9, %10 : f64
// CHECK-NEXT:       %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:       %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %15 = arith.mulf %11, %cst : f64
// CHECK-NEXT:       %16 = arith.addf %15, %14 : f64
// CHECK-NEXT:       "stencil.return"(%16) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[0,68]x[0,68]x[0,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[0,68]x[0,68]x[0,68]xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<?x?x?xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      "stencil.return"(%16) : (f64) -> ()
    }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: The stencil computation requires a field with lower bound at least #stencil.index<-1, -1, 0>, got #stencil.index<0, 0, 0>, min: #stencil.index<-1, -1, 0>

// -----

builtin.module {

  func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
      %6 = arith.addf %4, %5 : f64
      "stencil.return"(%6) : (f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:     %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:     %3 = "stencil.apply"(%0) ({
// CHECK-NEXT:     ^0(%4 : f64):
// CHECK-NEXT:       %5 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
// CHECK-NEXT:       %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:       "stencil.return"(%6) : (f64) -> ()
// CHECK-NEXT:     }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
// CHECK-NEXT:     "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
