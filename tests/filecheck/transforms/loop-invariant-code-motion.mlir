// RUN: xdsl-opt %s --split-input-file -p licm | filecheck %s

func.func public @ddot(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c8 step %c1 {
    %a0 = memref.load %arg0[%arg3] : memref<8xf64>
    %a1 = memref.load %arg1[%arg3] : memref<8xf64>
    %a2 = memref.load %arg2[] : memref<f64>
    %a3 = arith.mulf %a0, %a1 : f64
    %a4 = arith.addf %a2, %a3 : f64
    memref.store %a4, %arg2[] : memref<f64>
  }
  return %arg2 : memref<f64>
}
// CHECK:         func.func public @ddot(%arg0 : memref<8xf64>, %arg1 : memref<8xf64>, %arg2 : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %a0 = memref.load %arg0[%arg3] : memref<8xf64>
// CHECK-NEXT:        %a1 = memref.load %arg1[%arg3] : memref<8xf64>
// CHECK-NEXT:        %a2 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:        %a3 = arith.mulf %a0, %a1 : f64
// CHECK-NEXT:        %a4 = arith.addf %a2, %a3 : f64
// CHECK-NEXT:        memref.store %a4, %arg2[] : memref<f64>
// CHECK-NEXT:       }
// CHECK-NEXT:      func.return %arg2 : memref<f64>
// CHECK-NEXT:    }

//-----

func.func public @ot(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>) -> memref<f64> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c8 step %c1 {
    %a0 = memref.load %arg0[%arg3] : memref<8xf64>
    %a1 = memref.load %arg1[%arg3] : memref<8xf64>
    %a2 = memref.load %arg2[] : memref<f64>
    %a3 = arith.mulf %a0, %a1 : f64
    %a4 = arith.addf %a2, %a3 : f64
    memref.store %a4, %arg2[] : memref<f64>
  }
  return %arg2 : memref<f64>
}
// CHECK:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %a0 = memref.load %arg0[%arg3] : memref<8xf64>
// CHECK-NEXT:        %a1 = memref.load %arg1[%arg3] : memref<8xf64>
// CHECK-NEXT:        %a2 = memref.load %arg2[] : memref<f64>
// CHECK-NEXT:        %a3 = arith.mulf %a0, %a1 : f64
// CHECK-NEXT:        %a4 = arith.addf %a2, %a3 : f64
// CHECK-NEXT:        memref.store %a4, %arg2[] : memref<f64>
// CHECK-NEXT:       }
// CHECK-NEXT:        return %arg2 : memref<f64>
// CHECK-NEXT:      }

//-----

func.func @invariant_loop_dialect() {
  %ci0 = arith.constant 0 : index
  %ci10 = arith.constant 10 : index
  %ci1 = arith.constant 1 : index
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  scf.for %arg0 = %ci0 to %ci10 step %ci1 {
    scf.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: scf.for %arg0 = %ci0 to %ci10 step %ci1 {
  // CHECK-NEXT: scf.for %arg1 = %ci0 to %ci10 step %ci1 {
  // CHECK-NEXT: %v0 = arith.addf %cf7, %cf8 : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: }

  return
}
