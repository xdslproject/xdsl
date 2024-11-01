// RUN: xdsl-opt -p convert-memref-to-ptr,convert-ptr-to-riscv --print-op-generic --split-input-file %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic -pass-pipeline 'builtin.module(func.func(scf-for-loop-canonicalization,scf-for-loop-range-folding,scf-for-loop-canonicalization))' | xdsl-opt -p scf-for-loop-flatten --print-op-generic | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic -pass-pipeline 'builtin.module(func.func(scf-for-loop-canonicalization,scf-for-loop-range-folding,scf-for-loop-canonicalization))'

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

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "func.func"() <{function_type = (memref<10xi32>) -> (), sym_name = "fill"}> ({
// CHECK-NEXT:    ^bb0(%arg0: memref<10xi32>):
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 10 : index}> : () -> index
// CHECK-NEXT:      %3 = "arith.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:      %4 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:      %5 = "arith.muli"(%2, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
// CHECK-NEXT:      %6 = "arith.muli"(%3, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
// CHECK-NEXT:      "scf.for"(%1, %5, %6) ({
// CHECK-NEXT:      ^bb0(%arg1: index):
// CHECK-NEXT:        %7 = "builtin.unrealized_conversion_cast"(%arg0) : (memref<10xi32>) -> index
// CHECK-NEXT:        %8 = "arith.addi"(%7, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
// CHECK-NEXT:        %9 = "builtin.unrealized_conversion_cast"(%8) : (index) -> !riscv.reg
// CHECK-NEXT:        %10 = "builtin.unrealized_conversion_cast"(%4) : (i32) -> !riscv.reg
// CHECK-NEXT:        "riscv.sw"(%9, %10) {comment = "store int value to pointer", immediate = 0 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:        "scf.yield"() : () -> ()
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "func.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
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

// CHECK-NEXT:  "func.func"() <{function_type = (memref<10x10xi32>) -> (), sym_name = "fill2d"}> ({
// CHECK-NEXT:    ^bb0(%arg0: memref<10x10xi32>):
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 400 : index}> : () -> index
// CHECK-NEXT:      %3 = "arith.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT:      "scf.for"(%0, %2, %3) ({
// CHECK-NEXT:      ^bb0(%arg1: index):
// CHECK-NEXT:        %4 = "builtin.unrealized_conversion_cast"(%arg0) : (memref<10x10xi32>) -> index
// CHECK-NEXT:        %5 = "arith.addi"(%4, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
// CHECK-NEXT:        %6 = "builtin.unrealized_conversion_cast"(%5) : (index) -> !riscv.reg
// CHECK-NEXT:        %7 = "builtin.unrealized_conversion_cast"(%1) : (i32) -> !riscv.reg
// CHECK-NEXT:        "riscv.sw"(%6, %7) {comment = "store int value to pointer", immediate = 0 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:        "scf.yield"() : () -> ()
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "func.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
