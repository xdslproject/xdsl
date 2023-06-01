// RUN: xdsl-opt %s --split-input-file | filecheck %s

builtin.module {
  func.func @stencil_copy(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> f64
      %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
      "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @stencil_copy(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^0(%6 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func private @myfunc(%b0 : !stencil.field<?x?x?xf32>, %b1 : !stencil.field<?x?x?xf32>)  attributes {"param_names" = ["data"]}{
    %f0 = "stencil.cast"(%b0) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
    %f1 = "stencil.cast"(%b1) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index
    %fnp1, %fn = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
    ^0(%time : index, %fi : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, %fip1 : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>):
      %ti = "stencil.load"(%fi) : (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> !stencil.temp<?x?x?xf32>
      %tip1 = "stencil.apply"(%ti) ({
      ^1(%ti_ : !stencil.temp<?x?x?xf32>):
        %v = "stencil.access"(%ti) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf32>) -> f32
        "stencil.return"(%v) : (f32) -> ()
      }) : (!stencil.temp<?x?x?xf32>) -> !stencil.temp<?x?x?xf32>
      "stencil.store"(%tip1, %fip1) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<50, 80, 40>} : (!stencil.temp<?x?x?xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> ()
      "scf.yield"(%fip1, %fi) : (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> ()
    }) : (index, index, index, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>)
    "func.return"() : () -> ()
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func private @myfunc(%b0 : !stencil.field<?x?x?xf32>, %b1 : !stencil.field<?x?x?xf32>)  attributes {"param_names" = ["data"]}{
// CHECK-NEXT:     %f0 = "stencil.cast"(%b0) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:     %f1 = "stencil.cast"(%b1) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:     %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
// CHECK-NEXT:     %step = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %fnp1, %fn = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
// CHECK-NEXT:     ^0(%time : index, %fi : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, %fip1 : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>):
// CHECK-NEXT:       %ti = "stencil.load"(%fi) : (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> !stencil.temp<?x?x?xf32>
// CHECK-NEXT:       %tip1 = "stencil.apply"(%ti) ({
// CHECK-NEXT:       ^1(%ti_ : !stencil.temp<?x?x?xf32>):
// CHECK-NEXT:         %v = "stencil.access"(%ti) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf32>) -> f32
// CHECK-NEXT:         "stencil.return"(%v) : (f32) -> ()
// CHECK-NEXT:       }) : (!stencil.temp<?x?x?xf32>) -> !stencil.temp<?x?x?xf32>
// CHECK-NEXT:       "stencil.store"(%tip1, %fip1) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<50, 80, 40>} : (!stencil.temp<?x?x?xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> ()
// CHECK-NEXT:       "scf.yield"(%fip1, %fi) : (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> ()
// CHECK-NEXT:     }) : (index, index, index, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>)
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func private @stencil_laplace(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>) {
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<[-1,65]x[-1,65]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %16 = arith.mulf %11, %15 : f64
      %17 = arith.mulf %16, %13 : f64
      "stencil.return"(%17) : (f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> !stencil.temp<[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<64, 64>} : (!stencil.temp<[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @stencil_laplace(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^0(%6 : !stencil.temp<[-1,65]x[-1,65]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %12 = arith.addf %7, %8 : f64
// CHECK-NEXT:       %13 = arith.addf %9, %10 : f64
// CHECK-NEXT:       %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %16 = arith.mulf %11, %15 : f64
// CHECK-NEXT:       %17 = arith.mulf %16, %13 : f64
// CHECK-NEXT:       "stencil.return"(%17) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> !stencil.temp<[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<64, 64>} : (!stencil.temp<[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
