// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
  %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr<i32>
  %2 = "llvm.load"(%1) : (!llvm.ptr<i32>) -> i32
  %3 = "llvm.mlir.null"() : () -> !llvm.ptr
  %4 = "llvm.alloca"(%0) {"alignment" = 32 : i64, "elem_type" = i64} : (i64) -> !llvm.ptr
  %5 = "llvm.load"(%4) : (!llvm.ptr) -> i64
  %6 = "llvm.alloca"(%0) {"alignment" = 32 : i64} : (i64) -> !llvm.ptr<i64>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
// CHECK-NEXT:   %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:   %2 = "llvm.load"(%1) : (!llvm.ptr<i32>) -> i32
// CHECK-NEXT:   %3 = "llvm.mlir.null"() : () -> !llvm.ptr
// CHECK-NEXT:   %4 = "llvm.alloca"(%0) {"alignment" = 32 : i64, "elem_type" = i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:   %5 = "llvm.load"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:   %6 = "llvm.alloca"(%0) {"alignment" = 32 : i64} : (i64) -> !llvm.ptr<i64>
// CHECK-NEXT: }) : () -> ()
