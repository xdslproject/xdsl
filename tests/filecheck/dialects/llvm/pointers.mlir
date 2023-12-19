// RUN: XDSL_ROUNDTRIP
builtin.module {
  %0 = arith.constant 0 : i64
  %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr
  %2 = "llvm.load"(%1) : (!llvm.ptr) -> i32
  %3 = "llvm.mlir.null"() : () -> !llvm.ptr
  %4 = "llvm.alloca"(%0) {"alignment" = 32 : i64} : (i64) -> !llvm.ptr
  %5 = "llvm.load"(%4) : (!llvm.ptr) -> index
  %6 = "llvm.getelementptr"(%4, %0){elem_type = i32, rawConstantIndices = array<i32:-2147483648>} : (!llvm.ptr, i64) -> !llvm.ptr
  %7 = "llvm.alloca"(%0) : (i64) -> !llvm.ptr
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0 = arith.constant 0 : i64
// CHECK-NEXT:   %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr
// CHECK-NEXT:   %2 = "llvm.load"(%1) : (!llvm.ptr) -> i32
// CHECK-NEXT:   %3 = "llvm.mlir.null"() : () -> !llvm.ptr
// CHECK-NEXT:   %4 = "llvm.alloca"(%0) <{"alignment" = 32 : i64}> : (i64) -> !llvm.ptr
// CHECK-NEXT:   %5 = "llvm.load"(%4) : (!llvm.ptr) -> index
// CHECK-NEXT:   %6 = "llvm.getelementptr"(%4, %0) <{"elem_type" = i32, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:   %7 = "llvm.alloca"(%0) : (i64) -> !llvm.ptr
// CHECK-NEXT: }
