// RUN: xdsl-opt -p convert-memref-to-ptr,convert-ptr-type-offsets --print-op-generic --split-input-file %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic -pass-pipeline 'builtin.module(func.func(scf-for-loop-canonicalization,scf-for-loop-range-folding,scf-for-loop-canonicalization))' | xdsl-opt -p scf-for-loop-flatten --print-op-generic | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic -pass-pipeline 'builtin.module(func.func(scf-for-loop-canonicalization,scf-for-loop-range-folding,scf-for-loop-canonicalization))'

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

// CHECK:      "func.func"() <{function_type = (memref<10xi32>) -> (), sym_name = "fill"}> ({
// CHECK-NEXT:  ^bb0(%arg4: memref<10xi32>):
// CHECK-NEXT:    %12 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:    %13 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:    %14 = "arith.constant"() <{value = 40 : index}> : () -> index
// CHECK-NEXT:    %15 = "arith.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT:    "scf.for"(%12, %14, %15) ({
// CHECK-NEXT:    ^bb0(%arg5: index):
// CHECK-NEXT:      %16 = "ptr_xdsl.to_ptr"(%arg4) : (memref<10xi32>) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %17 = "ptr_xdsl.ptradd"(%16, %arg5) : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      "ptr_xdsl.store"(%17, %13) : (!ptr_xdsl.ptr, i32) -> ()
// CHECK-NEXT:      "scf.yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index) -> ()
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT:  }) : () -> ()


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

// CHECK-NEXT:   "func.func"() <{function_type = (memref<10x10xi32>) -> (), sym_name = "fill2d"}> ({
// CHECK-NEXT:   ^bb0(%arg2: memref<10x10xi32>):
// CHECK-NEXT:     %6 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:     %8 = "arith.constant"() <{value = 400 : index}> : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT:     "scf.for"(%6, %8, %9) ({
// CHECK-NEXT:     ^bb0(%arg3: index):
// CHECK-NEXT:       %10 = "ptr_xdsl.to_ptr"(%arg2) : (memref<10x10xi32>) -> !ptr_xdsl.ptr
// CHECK-NEXT:       %11 = "ptr_xdsl.ptradd"(%10, %arg3) : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:       "ptr_xdsl.store"(%11, %7) : (!ptr_xdsl.ptr, i32) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()


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

// CHECK-NEXT:   "func.func"() <{function_type = (memref<10x10x10xi32>) -> (), sym_name = "fill3d"}> ({
// CHECK-NEXT:   ^bb0(%arg0: memref<10x10x10xi32>):
// CHECK-NEXT:     %0 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %1 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:     %2 = "arith.constant"() <{value = 4000 : index}> : () -> index
// CHECK-NEXT:     %3 = "arith.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT:     "scf.for"(%0, %2, %3) ({
// CHECK-NEXT:     ^bb0(%arg1: index):
// CHECK-NEXT:       %4 = "ptr_xdsl.to_ptr"(%arg0) : (memref<10x10x10xi32>) -> !ptr_xdsl.ptr
// CHECK-NEXT:       %5 = "ptr_xdsl.ptradd"(%4, %arg1) : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:       "ptr_xdsl.store"(%5, %1) : (!ptr_xdsl.ptr, i32) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
