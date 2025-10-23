// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir | mlir-opt | filecheck %s

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
      %7 = stencil.access %6[-1, 0, 0] : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
      %8 = stencil.access %6[1, 0, 0] : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
      %9 = stencil.access %6[0, 1, 0] : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
      %10 = stencil.access %6[0, -1, 0] : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
      %11 = stencil.access %6[0, 0, 0] : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      stencil.return %16 : f64
    }
    stencil.store %5 to %3 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }
}

// CHECK:       module {
// CHECK-NEXT:    func.func @stencil_hdiff(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) {
// CHECK-NEXT:      %cast = memref.cast %arg0 : memref<?x?x?xf64> to memref<72x72x72xf64>
// CHECK-NEXT:      %cast_0 = memref.cast %arg1 : memref<?x?x?xf64> to memref<72x72x72xf64>
// CHECK-NEXT:      %subview = memref.subview %cast_0[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %subview_1 = memref.subview %cast[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c0_2 = arith.constant 0 : index
// CHECK-NEXT:      %c0_3 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c1_4 = arith.constant 1 : index
// CHECK-NEXT:      %c1_5 = arith.constant 1 : index
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %c64_6 = arith.constant 64 : index
// CHECK-NEXT:      %c64_7 = arith.constant 64 : index
// CHECK-NEXT:      scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0_2, %c0_3) to (%c64, %c64_6, %c64_7) step (%c1, %c1_4, %c1_5) {
// CHECK-NEXT:        %c-1 = arith.constant -1 : index
// CHECK-NEXT:        %0 = arith.addi %arg2, %c-1 : index
// CHECK-NEXT:        %1 = memref.load %subview_1[%0, %arg3, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %c1_8 = arith.constant 1 : index
// CHECK-NEXT:        %2 = arith.addi %arg2, %c1_8 : index
// CHECK-NEXT:        %3 = memref.load %subview_1[%2, %arg3, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %c1_9 = arith.constant 1 : index
// CHECK-NEXT:        %4 = arith.addi %arg3, %c1_9 : index
// CHECK-NEXT:        %5 = memref.load %subview_1[%arg2, %4, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %c-1_10 = arith.constant -1 : index
// CHECK-NEXT:        %6 = arith.addi %arg3, %c-1_10 : index
// CHECK-NEXT:        %7 = memref.load %subview_1[%arg2, %6, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %8 = memref.load %subview_1[%arg2, %arg3, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %9 = arith.addf %1, %3 : f64
// CHECK-NEXT:        %10 = arith.addf %5, %7 : f64
// CHECK-NEXT:        %11 = arith.addf %9, %10 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %12 = arith.mulf %8, %cst : f64
// CHECK-NEXT:        %13 = arith.addf %12, %11 : f64
// CHECK-NEXT:        memref.store %13, %subview[%arg2, %arg3, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
