// RUN: xdsl-opt -p convert-memref-to-ptr,convert-ptr-type-offsets,mlir-opt[scf-for-loop-canonicalization,scf-for-loop-range-folding,scf-for-loop-canonicalization],scf-for-loop-flatten,mlir-opt[scf-for-loop-canonicalization,scf-for-loop-range-folding,scf-for-loop-canonicalization] --split-input-file %s | filecheck %s

func.func @fill(%m: memref<10xi32>) {
    %c0 = arith.constant 0 : index
    %end = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %val = arith.constant 100 : i32
    scf.for %i = %c0 to %end step %c1 {
        memref.store %val, %m[%i] : memref<10xi32>
    }
    return
}

// CHECK:       func.func @fill(%arg4 : memref<10xi32>) {
// CHECK-NEXT:    %0 = arith.constant 0 : index
// CHECK-NEXT:    %1 = arith.constant 100 : i32
// CHECK-NEXT:    %2 = arith.constant 40 : index
// CHECK-NEXT:    %3 = arith.constant 4 : index
// CHECK-NEXT:    scf.for %arg5 = %0 to %2 step %3 {
// CHECK-NEXT:      %4 = ptr_xdsl.to_ptr %arg4 : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %5 = ptr_xdsl.ptradd %4, %arg5 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      ptr_xdsl.store %1, %5 : i32, !ptr_xdsl.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

func.func @fill2d(%m: memref<10x10xi32>) {
    %c0 = arith.constant 0 : index
    %end = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %val = arith.constant 100 : i32
    scf.for %i = %c0 to %end step %c1 {
        scf.for %j = %c0 to %end step %c1 {
            memref.store %val, %m[%i, %j] : memref<10x10xi32>
        }
    }
    return
}

// CHECK-NEXT:  func.func @fill2d(%arg2 : memref<10x10xi32>) {
// CHECK-NEXT:    %0 = arith.constant 0 : index
// CHECK-NEXT:    %1 = arith.constant 100 : i32
// CHECK-NEXT:    %2 = arith.constant 400 : index
// CHECK-NEXT:    %3 = arith.constant 4 : index
// CHECK-NEXT:    scf.for %arg3 = %0 to %2 step %3 {
// CHECK-NEXT:      %4 = ptr_xdsl.to_ptr %arg2 : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %5 = ptr_xdsl.ptradd %4, %arg3 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      ptr_xdsl.store %1, %5 : i32, !ptr_xdsl.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

func.func @fill3d(%m: memref<10x10x10xi32>) {
    %c0 = arith.constant 0 : index
    %end = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %val = arith.constant 100 : i32
    scf.for %i = %c0 to %end step %c1 {
        scf.for %j = %c0 to %end step %c1 {
            scf.for %k = %c0 to %end step %c1 {
                memref.store %val, %m[%i, %j, %k] : memref<10x10x10xi32>
            }
        }
    }
    return
}

// CHECK-NEXT:  func.func @fill3d(%arg0 : memref<10x10x10xi32>) {
// CHECK-NEXT:    %0 = arith.constant 0 : index
// CHECK-NEXT:    %1 = arith.constant 100 : i32
// CHECK-NEXT:    %2 = arith.constant 4000 : index
// CHECK-NEXT:    %3 = arith.constant 4 : index
// CHECK-NEXT:    scf.for %arg1 = %0 to %2 step %3 {
// CHECK-NEXT:      %4 = ptr_xdsl.to_ptr %arg0 : memref<10x10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %5 = ptr_xdsl.ptradd %4, %arg1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      ptr_xdsl.store %1, %5 : i32, !ptr_xdsl.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
