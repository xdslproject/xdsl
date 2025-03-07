// RUN: xdsl-opt -p xdsl-opt -p convert-memref-to-ptr,convert-ptr-type-offsets,canonicalize,scf-for-loop-range-folding,canonicalize,scf-for-loop-flatten,canonicalize,scf-for-loop-range-folding,canonicalize %s

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

// CHECK:        func.func @fill(%m : memref<10xi32>) {
// CHECK-NEXT:     %val = arith.constant 100 : i32
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = arith.constant 40 : index
// CHECK-NEXT:     %2 = arith.constant 4 : index
// CHECK-NEXT:     scf.for %i = %0 to %1 step %2 {
// CHECK-NEXT:       %3 = ptr_xdsl.to_ptr %m : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:       %offset_pointer = ptr_xdsl.ptradd %3, %i : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:       ptr_xdsl.store %val, %offset_pointer : i32, !ptr_xdsl.ptr
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


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

// CHECK-NEXT:   func.func @fill2d(%m : memref<10x10xi32>) {
// CHECK-NEXT:     %val = arith.constant 100 : i32
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = arith.constant 400 : index
// CHECK-NEXT:     %2 = arith.constant 4 : index
// CHECK-NEXT:     scf.for %j = %0 to %1 step %2 {
// CHECK-NEXT:       %3 = ptr_xdsl.to_ptr %m : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:       %offset_pointer = ptr_xdsl.ptradd %3, %j : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:       ptr_xdsl.store %val, %offset_pointer : i32, !ptr_xdsl.ptr
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


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

// CHECK-NEXT:    func.func @fill3d(%m : memref<10x10x10xi32>) {
// CHECK-NEXT:      %val = arith.constant 100 : i32
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 4000 : index
// CHECK-NEXT:      %2 = arith.constant 4 : index
// CHECK-NEXT:      scf.for %k = %0 to %1 step %2 {
// CHECK-NEXT:        %3 = ptr_xdsl.to_ptr %m : memref<10x10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:        %offset_pointer = ptr_xdsl.ptradd %3, %k : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        ptr_xdsl.store %val, %offset_pointer : i32, !ptr_xdsl.ptr
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
