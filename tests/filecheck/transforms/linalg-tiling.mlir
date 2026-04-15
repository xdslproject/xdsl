// RUN: xdsl-opt %s -p "linalg-tiling{tile-sizes=2,2}" | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>)

linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%A, %B : memref<4x4xf64>, memref<4x4xf64>) outs(%C : memref<4x4xf64>) {
^bb0(%a: f64, %b: f64, %c: f64):
    %sum = arith.addf %a, %b : f64
    linalg.yield %sum : f64
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %5 = %0 to %1 step %3 {
// CHECK-NEXT:     scf.for %6 = %0 to %2 step %4 {
// CHECK-NEXT:       %7 = memref.subview %A[%5, %6] [2, 2] [1, 1] : memref<4x4xf64> to memref<2x2xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       %8 = memref.subview %B[%5, %6] [2, 2] [1, 1] : memref<4x4xf64> to memref<2x2xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       %9 = memref.subview %C[%5, %6] [2, 2] [1, 1] : memref<4x4xf64> to memref<2x2xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : memref<2x2xf64, strided<[4, 1], offset: ?>>, memref<2x2xf64, strided<[4, 1], offset: ?>>) outs(%9 : memref<2x2xf64, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:       ^bb0(%a: f64, %b: f64, %c: f64):
// CHECK-NEXT:         %sum = arith.addf %a, %b : f64
// CHECK-NEXT:         linalg.yield %sum : f64
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
