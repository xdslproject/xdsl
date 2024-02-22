// RUN: xdsl-opt %s --split-input-file -p loop-hoist-memref | filecheck %s

func.func public @ddot(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c8 step %c1 {
    %0 = memref.load %arg0[%arg3] : memref<8xf64>
    %1 = memref.load %arg1[%arg3] : memref<8xf64>
    %2 = memref.load %arg2[] : memref<f64>
    %3 = arith.mulf %0, %1 : f64
    %4 = arith.addf %2, %3 : f64
    memref.store %4, %arg2[] : memref<f64>
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @ddot(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %0 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:      %1 = scf.for %2 = %c0 to %c8 step %c1 iter_args(%3 = %0) -> (f64) {
// CHECK-NEXT:        %4 = memref.load %arg0[%2] : memref<8xf64>
// CHECK-NEXT:        %5 = memref.load %arg1[%2] : memref<8xf64>
// CHECK-NEXT:        %6 = arith.mulf %4, %5 : f64
// CHECK-NEXT:        %7 = arith.addf %3, %6 : f64
// CHECK-NEXT:        scf.yield %7 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1, %arg2[] : memref<f64>
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

// -----

func.func public @repeat_ddot(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c8 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<8xf64>
      %1 = memref.load %arg1[%arg3] : memref<8xf64>
      %2 = memref.load %arg2[] : memref<f64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %2, %3 : f64
      memref.store %4, %arg2[] : memref<f64>
    }
  }
  return %arg2 : memref<f64>
}

// CHECK:         func.func public @repeat_ddot(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %0 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:      %1 = scf.for %2 = %c0 to %c8 step %c1 iter_args(%3 = %0) -> (f64) {
// CHECK-NEXT:        %4 = scf.for %5 = %c0 to %c8 step %c1 iter_args(%6 = %3) -> (f64) {
// CHECK-NEXT:          %7 = memref.load %arg0[%2] : memref<8xf64>
// CHECK-NEXT:          %8 = memref.load %arg1[%2] : memref<8xf64>
// CHECK-NEXT:          %9 = arith.mulf %7, %8 : f64
// CHECK-NEXT:          %10 = arith.addf %6, %9 : f64
// CHECK-NEXT:          scf.yield %10 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %4 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1, %arg2[] : memref<f64>
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

// -----

func.func public @matmul(%arg0: memref<8x8xf64> {llvm.noalias}, %arg1: memref<8x8xf64> {llvm.noalias}, %arg2: memref<8x8xf64> {llvm.noalias}) -> memref<8x8xf64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c8 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c8 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg5] : memref<8x8xf64>
        %1 = memref.load %arg1[%arg5, %arg4] : memref<8x8xf64>
        %2 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf64>
        %3 = arith.mulf %0, %1 : f64
        %4 = arith.addf %2, %3 : f64
        memref.store %4, %arg2[%arg3, %arg4] : memref<8x8xf64>
      }
    }
  }
  return %arg2 : memref<8x8xf64>
}

// CHECK:         func.func public @matmul(%arg0 : memref<8x8xf64> {"llvm.noalias"}, %arg1 : memref<8x8xf64> {"llvm.noalias"}, %arg2 : memref<8x8xf64> {"llvm.noalias"}) -> memref<8x8xf64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c8 step %c1 {
// CHECK-NEXT:        scf.for %arg4 = %c0 to %c8 step %c1 {
// CHECK-NEXT:          %0 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf64>
// CHECK-NEXT:          %1 = scf.for %2 = %c0 to %c8 step %c1 iter_args(%3 = %0) -> (f64) {
// CHECK-NEXT:            %4 = memref.load %arg0[%arg3, %2] : memref<8x8xf64>
// CHECK-NEXT:            %5 = memref.load %arg1[%2, %arg4] : memref<8x8xf64>
// CHECK-NEXT:            %6 = arith.mulf %4, %5 : f64
// CHECK-NEXT:            %7 = arith.addf %3, %6 : f64
// CHECK-NEXT:            scf.yield %7 : f64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %1, %arg2[%arg3, %arg4] : memref<8x8xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %arg2 : memref<8x8xf64>
// CHECK-NEXT:    }
