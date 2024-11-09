// RUN: xdsl-opt -p scf-for-loop-range-folding --split-input-file %s | filecheck %s

func.func @fold_one_loop(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = arith.addi %arg2, %i : index
    %1 = arith.muli %0, %c4 : index
    %2 = memref.load %arg0[%1] : memref<?xi32>
    %3 = arith.muli %2, %2 : i32
    memref.store %3, %arg0[%1] : memref<?xi32>
  }
  return
}

// CHECK:       %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %c4 = arith.constant 4 : index
// CHECK-NEXT:  %0 = arith.addi %c0, %arg2 : index
// CHECK-NEXT:  %1 = arith.addi %arg1, %arg2 : index
// CHECK-NEXT:  %2 = arith.muli %0, %c4 : index
// CHECK-NEXT:  %3 = arith.muli %1, %c4 : index
// CHECK-NEXT:  %4 = arith.muli %c1, %c4 : index
// CHECK-NEXT:  scf.for %i = %2 to %3 step %4 {
// CHECK-NEXT:      %5 = memref.load %arg0[%i] : memref<?xi32>
// CHECK-NEXT:      %6 = arith.muli %5, %5 : i32
// CHECK-NEXT:      memref.store %6, %arg0[%i] : memref<?xi32>
// CHECK-NEXT:  }

// In this example muli can't be taken out of the loop
func.func @fold_only_first_add(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = arith.addi %arg2, %i : index
    %1 = arith.addi %arg2, %c4 : index
    %2 = arith.muli %0, %1 : index
    %3 = memref.load %arg0[%2] : memref<?xi32>
    %4 = arith.muli %3, %3 : i32
    memref.store %4, %arg0[%2] : memref<?xi32>
  }
  return
}

// CHECK:         %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %0 = arith.addi %c0, %arg2 : index
// CHECK-NEXT:    %1 = arith.addi %arg1, %arg2 : index
// CHECK-NEXT:    scf.for %i = %0 to %1 step %c1 {
// CHECK-NEXT:      %2 = arith.addi %arg2, %c4 : index
// CHECK-NEXT:      %3 = arith.muli %i, %2 : index
// CHECK-NEXT:      %4 = memref.load %arg0[%3] : memref<?xi32>
// CHECK-NEXT:      %5 = arith.muli %4, %4 : i32
// CHECK-NEXT:      memref.store %5, %arg0[%3] : memref<?xi32>
// CHECK-NEXT:    }

func.func @fold_two_loops(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  scf.for %j = %c0 to %c10 step %c1 {
    scf.for %i = %j to %arg1 step %c1 {
      %0 = arith.addi %arg2, %i : index
      %1 = arith.muli %0, %c4 : index
      %2 = memref.load %arg0[%1] : memref<?xi32>
      %3 = arith.muli %2, %2 : i32
      memref.store %3, %arg0[%1] : memref<?xi32>
    }
  }
  return
}

// CHECK:         %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %0 = arith.addi %c0, %arg2 : index
// CHECK-NEXT:    %1 = arith.addi %c10, %arg2 : index
// CHECK-NEXT:    %2 = arith.muli %0, %c4 : index
// CHECK-NEXT:    %3 = arith.muli %1, %c4 : index
// CHECK-NEXT:    %4 = arith.muli %c1, %c4 : index
// CHECK-NEXT:    scf.for %j = %2 to %3 step %4 {
// CHECK-NEXT:      %5 = arith.addi %arg1, %arg2 : index
// CHECK-NEXT:      %6 = arith.muli %5, %c4 : index
// CHECK-NEXT:      %7 = arith.muli %c1, %c4 : index
// CHECK-NEXT:      scf.for %i = %j to %6 step %7 {
// CHECK-NEXT:        %8 = memref.load %arg0[%i] : memref<?xi32>
// CHECK-NEXT:        %9 = arith.muli %8, %8 : i32
// CHECK-NEXT:        memref.store %9, %arg0[%i] : memref<?xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
