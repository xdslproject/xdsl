// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir | mlir-opt | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
    %3 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %6 = "stencil.load"(%3) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<1, 0, 0>} :  (!stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<0, 1, 0>} :  (!stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<0, 0, 0>} :  (!stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (f64) -> ()
    }) : (!stencil.temp<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()

// CHECK:       module {
// CHECK-NEXT:    func.func @stencil_hdiff(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) {
// CHECK-NEXT:      %cast = memref.cast %arg0 : memref<?x?x?xf64> to memref<72x72x72xf64>
// CHECK-NEXT:      %cast_0 = memref.cast %arg1 : memref<?x?x?xf64> to memref<72x72x72xf64>
// CHECK-NEXT:      %subview = memref.subview %cast_0[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %subview_1 = memref.subview %cast[4, 4, 4] [72, 72, 72] [1, 1, 1] : memref<72x72x72xf64> to memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c0_2 = arith.constant 0 : index
// CHECK-NEXT:      %c0_3 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c1_4 = arith.constant 1 : index
// CHECK-NEXT:      %c1_5 = arith.constant 1 : index
// CHECK-NEXT:      %c1_6 = arith.constant 1 : index
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %c64_7 = arith.constant 64 : index
// CHECK-NEXT:      %c64_8 = arith.constant 64 : index
// CHECK-NEXT:      scf.parallel (%arg2) = (%c0) to (%c64) step (%c1_4) {
// CHECK-NEXT:        scf.for %arg3 = %c0_2 to %c64_7 step %c1_5 {
// CHECK-NEXT:          scf.for %arg4 = %c0_3 to %c64_8 step %c1_6 {
// CHECK-NEXT:            %c-1 = arith.constant -1 : index
// CHECK-NEXT:            %0 = arith.addi %arg2, %c-1 : index
// CHECK-NEXT:            %1 = memref.load %subview_1[%0, %arg3, %arg4] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %c1_9 = arith.constant 1 : index
// CHECK-NEXT:            %2 = arith.addi %arg2, %c1_9 : index
// CHECK-NEXT:            %3 = memref.load %subview_1[%2, %arg3, %arg4] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %c1_10 = arith.constant 1 : index
// CHECK-NEXT:            %4 = arith.addi %arg3, %c1_10 : index
// CHECK-NEXT:            %5 = memref.load %subview_1[%arg2, %4, %arg4] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %c-1_11 = arith.constant -1 : index
// CHECK-NEXT:            %6 = arith.addi %arg3, %c-1_11 : index
// CHECK-NEXT:            %7 = memref.load %subview_1[%arg2, %6, %arg4] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %8 = memref.load %subview_1[%arg2, %arg3, %arg4] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %9 = arith.addf %1, %3 : f64
// CHECK-NEXT:            %10 = arith.addf %5, %7 : f64
// CHECK-NEXT:            %11 = arith.addf %9, %10 : f64
// CHECK-NEXT:            %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:            %12 = arith.mulf %8, %cst : f64
// CHECK-NEXT:            %13 = arith.addf %12, %11 : f64
// CHECK-NEXT:            memref.store %13, %subview[%arg2, %arg3, %arg4] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
