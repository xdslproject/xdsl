// RUN: xdsl-opt %s -p loop-hoist-memref | filecheck %s

module {
  func.func public @foo(%arg0: memref<128xf64>, %arg1: memref<128xf64>, %arg2: memref<f64>) -> memref<f64> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c128 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<128xf64>
      %1 = memref.load %arg1[%arg3] : memref<128xf64>
      %2 = memref.load %arg2[] : memref<f64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %2, %3 : f64
      memref.store %4, %arg2[] : memref<f64>
    }
    return %arg2 : memref<f64>
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func public @foo(%arg0 : memref<128xf64>, %arg1 : memref<128xf64>, %arg2 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c128 = arith.constant 128 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %0 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:      %1 = scf.for %2 = %c0 to %c128 step %c1 iter_args(%3 = %0) -> (f64) {
// CHECK-NEXT:        %4 = memref.load %arg0[%2] : memref<128xf64>
// CHECK-NEXT:        %5 = memref.load %arg1[%2] : memref<128xf64>
// CHECK-NEXT:        %6 = arith.mulf %4, %5 : f64
// CHECK-NEXT:        %7 = arith.addf %3, %6 : f64
// CHECK-NEXT:        scf.yield %7 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1, %arg2[] : memref<f64>
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
