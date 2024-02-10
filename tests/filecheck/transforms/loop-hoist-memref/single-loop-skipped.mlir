// RUN: xdsl-opt %s --split-input-file -p loop-hoist-memref | filecheck %s

// skip loads from the same location
func.func public @foo(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>, %arg3: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %idx = %c0 to %c8 step %c1 {
    %0 = memref.load %arg0[%idx] : memref<8xf64>
    %1 = memref.load %arg1[%idx] : memref<8xf64>
    %2 = memref.load %arg3[] : memref<f64>
    %3 = memref.load %arg3[] : memref<f64>
    %4 = arith.mulf %0, %1 : f64
    %5 = arith.addf %3, %4 : f64
    memref.store %5, %arg3[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @foo(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>, %arg3 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %idx = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %0 = memref.load %arg0[%idx] : memref<8xf64>
// CHECK-NEXT:        %1 = memref.load %arg1[%idx] : memref<8xf64>
// CHECK-NEXT:        %2 = memref.load %arg3[] : memref<f64>
// CHECK-NEXT:        %3 = memref.load %arg3[] : memref<f64>
// CHECK-NEXT:        %4 = arith.mulf %0, %1 : f64
// CHECK-NEXT:        %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:        memref.store %5, %arg3[] : memref<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

// -----

// skip stores using the same value
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
    memref.store %5, %arg2[] : memref<f64>
    memref.store %5, %arg3[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @foo(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>, %arg3 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %idx = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %0 = memref.load %arg0[%idx] : memref<8xf64>
// CHECK-NEXT:        %1 = memref.load %arg1[%idx] : memref<8xf64>
// CHECK-NEXT:        %2 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:        %3 = memref.load %arg3[] : memref<f64>
// CHECK-NEXT:        %4 = arith.mulf %0, %1 : f64
// CHECK-NEXT:        %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:        memref.store %5, %arg2[] : memref<f64>
// CHECK-NEXT:        memref.store %5, %arg3[] : memref<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

// -----

// skip stores using the same location
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
    memref.store %4, %arg3[] : memref<f64>
    memref.store %5, %arg3[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @foo(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>, %arg3 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %idx = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %0 = memref.load %arg0[%idx] : memref<8xf64>
// CHECK-NEXT:        %1 = memref.load %arg1[%idx] : memref<8xf64>
// CHECK-NEXT:        %2 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:        %3 = memref.load %arg3[] : memref<f64>
// CHECK-NEXT:        %4 = arith.mulf %0, %1 : f64
// CHECK-NEXT:        %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:        memref.store %4, %arg3[] : memref<f64>
// CHECK-NEXT:        memref.store %5, %arg3[] : memref<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }
