// RUN: xdsl-opt %s -t mlir -p convert-stencil-to-ll-mlir | mlir-opt | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %3 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %4 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %6 = "stencil.load"(%3) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, -1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (!stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0: i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()

// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @stencil_hdiff(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) {
// CHECK-NEXT:     %cast = memref.cast %arg0 : memref<?x?x?xf64> to memref<72x72x72xf64>
// CHECK-NEXT:     %cast_0 = memref.cast %arg1 : memref<?x?x?xf64> to memref<72x72x72xf64>
// CHECK-NEXT:     %subview = memref.subview %cast_0[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %subview_1 = memref.subview %cast[0, 0, 0] [72, 72, 72] [1, 1, 1] : memref<72x72x72xf64> to memref<72x72x72xf64, strided<[5184, 72, 1]>>
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c64 = arith.constant 64 : index
// CHECK-NEXT:     %c64_2 = arith.constant 64 : index
// CHECK-NEXT:     %c64_3 = arith.constant 64 : index
// CHECK-NEXT:     scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c64, %c64_2, %c64_3) step (%c1, %c1, %c1) {
// CHECK-NEXT:       %c3 = arith.constant 3 : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %c4_4 = arith.constant 4 : index
// CHECK-NEXT:       %0 = arith.addi %arg2, %c3 : index
// CHECK-NEXT:       %1 = arith.addi %arg3, %c4 : index
// CHECK-NEXT:       %2 = arith.addi %arg4, %c4_4 : index
// CHECK-NEXT:       %3 = memref.load %subview_1[%0, %1, %2] : memref<72x72x72xf64, strided<[5184, 72, 1]>>
// CHECK-NEXT:       %c5 = arith.constant 5 : index
// CHECK-NEXT:       %c4_5 = arith.constant 4 : index
// CHECK-NEXT:       %c4_6 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.addi %arg2, %c5 : index
// CHECK-NEXT:       %5 = arith.addi %arg3, %c4_5 : index
// CHECK-NEXT:       %6 = arith.addi %arg4, %c4_6 : index
// CHECK-NEXT:       %7 = memref.load %subview_1[%4, %5, %6] : memref<72x72x72xf64, strided<[5184, 72, 1]>>
// CHECK-NEXT:       %c4_7 = arith.constant 4 : index
// CHECK-NEXT:       %c5_8 = arith.constant 5 : index
// CHECK-NEXT:       %c4_9 = arith.constant 4 : index
// CHECK-NEXT:       %8 = arith.addi %arg2, %c4_7 : index
// CHECK-NEXT:       %9 = arith.addi %arg3, %c5_8 : index
// CHECK-NEXT:       %10 = arith.addi %arg4, %c4_9 : index
// CHECK-NEXT:       %11 = memref.load %subview_1[%8, %9, %10] : memref<72x72x72xf64, strided<[5184, 72, 1]>>
// CHECK-NEXT:       %c4_10 = arith.constant 4 : index
// CHECK-NEXT:       %c3_11 = arith.constant 3 : index
// CHECK-NEXT:       %c4_12 = arith.constant 4 : index
// CHECK-NEXT:       %12 = arith.addi %arg2, %c4_10 : index
// CHECK-NEXT:       %13 = arith.addi %arg3, %c3_11 : index
// CHECK-NEXT:       %14 = arith.addi %arg4, %c4_12 : index
// CHECK-NEXT:       %15 = memref.load %subview_1[%12, %13, %14] : memref<72x72x72xf64, strided<[5184, 72, 1]>>
// CHECK-NEXT:       %c4_13 = arith.constant 4 : index
// CHECK-NEXT:       %c4_14 = arith.constant 4 : index
// CHECK-NEXT:       %c4_15 = arith.constant 4 : index
// CHECK-NEXT:       %16 = arith.addi %arg2, %c4_13 : index
// CHECK-NEXT:       %17 = arith.addi %arg3, %c4_14 : index
// CHECK-NEXT:       %18 = arith.addi %arg4, %c4_15 : index
// CHECK-NEXT:       %19 = memref.load %subview_1[%16, %17, %18] : memref<72x72x72xf64, strided<[5184, 72, 1]>>
// CHECK-NEXT:       %20 = arith.addf %3, %7 : f64
// CHECK-NEXT:       %21 = arith.addf %11, %15 : f64
// CHECK-NEXT:       %22 = arith.addf %20, %21 : f64
// CHECK-NEXT:       %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:       %23 = arith.mulf %19, %cst : f64
// CHECK-NEXT:       %24 = arith.addf %23, %22 : f64
// CHECK-NEXT:       memref.store %24, %subview[%arg2, %arg3, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
