// RUN: XDSL_ROUNDTRIP
builtin.module {
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<13 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "Hello world!\n", "unnamed_addr" = 0 : i64} : () -> ()
  %0 = "llvm.mlir.addressof"() {"global_name" = @str0} : () -> !llvm.ptr
  %1 = "llvm.getelementptr"(%0) <{"elem_type" = !llvm.array<13 x i8>, noWrapFlags = 0 : i32, "rawConstantIndices" = array<i32: 0, 0>}> : (!llvm.ptr) -> !llvm.ptr
}

// CHECK: builtin.module {
// CHECK-NEXT:   "llvm.mlir.global"() <{global_type = !llvm.array<13 x i8>, constant, sym_name = "str0", linkage = #llvm.linkage<"internal">, value = "Hello world!\n", addr_space = 0 : i32, unnamed_addr = 0 : i64}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   %0 = "llvm.mlir.addressof"() <{global_name = @str0}> : () -> !llvm.ptr
// CHECK-NEXT:   %1 = "llvm.getelementptr"(%0) <{elem_type = !llvm.array<13 x i8>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, 0>}> : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: }
