// RUN: xdsl-opt %s -t mlir | filecheck %s
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
  %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr<i32>
  %2 = "llvm.load"(%1) : (!llvm.ptr<i32>) -> i32
  %3 = "llvm.mlir.null"() : () -> !llvm.ptr
  %4 = "llvm.alloca"(%0) {"alignment" = 32 : i64} : (i64) -> !llvm.ptr<index>
  %5 = "llvm.load"(%4) : (!llvm.ptr<index>) -> index
  %6 = "llvm.getelementptr"(%4, %0){elem_type = i32, rawConstantIndices = array<i32:-2147483648>} : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
// CHECK-NEXT:   %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:   %2 = "llvm.load"(%1) : (!llvm.ptr<i32>) -> i32
// CHECK-NEXT:   %3 = "llvm.mlir.null"() : () -> !llvm.ptr
// CHECK-NEXT:   %4 = "llvm.alloca"(%0) {"alignment" = 32 : i64} : (i64) -> !llvm.ptr<index>
// CHECK-NEXT:   %5 = "llvm.load"(%4) : (!llvm.ptr<index>) -> index
// CHECK-NEXT:   %6 = "llvm.getelementptr"(%4, %0) {"elem_type" = i32, "rawConstantIndices" = array<i32: -2147483648>} : (!llvm.ptr<index>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT: }) : () -> ()
