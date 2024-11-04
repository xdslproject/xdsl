// RUN: xdsl-opt %s --split-input-file -p loop-hoist-memref | filecheck %s

// single load-store pair hoisting
func.func public @foo(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %idx = %c0 to %c8 step %c1 {
    %0 = memref.load %arg0[%idx] : memref<8xf64>
    %1 = memref.load %arg1[%idx] : memref<8xf64>
    %2 = memref.load %arg2[] : memref<f64>
    %3 = arith.mulf %0, %1 : f64
    %4 = arith.addf %2, %3 : f64
    memref.store %4, %arg2[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:       func.func public @foo(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %0 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:      %1 = scf.for %idx = %c0 to %c8 step %c1 iter_args(%2 = %0) -> (f64) {
// CHECK-NEXT:        %3 = memref.load %arg0[%idx] : memref<8xf64>
// CHECK-NEXT:        %4 = memref.load %arg1[%idx] : memref<8xf64>
// CHECK-NEXT:        %5 = arith.mulf %3, %4 : f64
// CHECK-NEXT:        %6 = arith.addf %2, %5 : f64
// CHECK-NEXT:        scf.yield %6 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1, %arg2[] : memref<f64>
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

// -----

// multiple non-overlapping load-store pair hoisting
func.func public @foo(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>, %arg3: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %idx = %c0 to %c8 step %c1 {
    %0 = memref.load %arg0[%idx] : memref<8xf64>
    %1 = memref.load %arg1[%idx] : memref<8xf64>
    %2 = memref.load %arg2[] : memref<f64>
    %3 = memref.load %arg3[] : memref<f64>
    %4 = arith.mulf %0, %1 : f64
    %5 = arith.addf %3, %4 : f64
    %6 = arith.addf %2, %5 : f64
    memref.store %4, %arg2[] : memref<f64>
    memref.store %5, %arg3[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @foo(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>, %arg3 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %0 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:      %1 = memref.load %arg3[] : memref<f64>
// CHECK-NEXT:      %2, %3 = scf.for %idx = %c0 to %c8 step %c1 iter_args(%4 = %0, %5 = %1) -> (f64, f64) {
// CHECK-NEXT:        %6 = memref.load %arg0[%idx] : memref<8xf64>
// CHECK-NEXT:        %7 = memref.load %arg1[%idx] : memref<8xf64>
// CHECK-NEXT:        %8 = arith.mulf %6, %7 : f64
// CHECK-NEXT:        %9 = arith.addf %5, %8 : f64
// CHECK-NEXT:        %10 = arith.addf %4, %9 : f64
// CHECK-NEXT:        scf.yield %8, %9 : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %2, %arg2[] : memref<f64>
// CHECK-NEXT:      memref.store %3, %arg3[] : memref<f64>
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

// -----

// multiple overlapping load-store pair hoisting
func.func public @foo(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>, %arg3: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %idx = %c0 to %c8 step %c1 {
    %0 = memref.load %arg0[%idx] : memref<8xf64>
    %1 = memref.load %arg1[%idx] : memref<8xf64>
    %2 = memref.load %arg2[] : memref<f64>
    %3 = memref.load %arg3[] : memref<f64>
    %4 = arith.mulf %0, %1 : f64
    %5 = arith.addf %3, %4 : f64
    %6 = arith.addf %2, %5 : f64
    memref.store %5, %arg3[] : memref<f64>
    memref.store %4, %arg2[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @foo(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>, %arg3 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %0 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:      %1 = memref.load %arg3[] : memref<f64>
// CHECK-NEXT:      %2, %3 = scf.for %idx = %c0 to %c8 step %c1 iter_args(%4 = %0, %5 = %1) -> (f64, f64) {
// CHECK-NEXT:        %6 = memref.load %arg0[%idx] : memref<8xf64>
// CHECK-NEXT:        %7 = memref.load %arg1[%idx] : memref<8xf64>
// CHECK-NEXT:        %8 = arith.mulf %6, %7 : f64
// CHECK-NEXT:        %9 = arith.addf %5, %8 : f64
// CHECK-NEXT:        %10 = arith.addf %4, %9 : f64
// CHECK-NEXT:        scf.yield %8, %9 : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %2, %arg2[] : memref<f64>
// CHECK-NEXT:      memref.store %3, %arg3[] : memref<f64>
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }
