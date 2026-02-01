// RUN: XDSL_ROUNDTRIP
builtin.module {
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<13 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "Hello world!\n", "unnamed_addr" = 0 : i64} : () -> ()
  %0 = "llvm.mlir.addressof"() {"global_name" = @str0} : () -> !llvm.ptr
  %1 = "llvm.getelementptr"(%0) <{"elem_type" = !llvm.array<13 x i8>, noWrapFlags = 0 : i32, "rawConstantIndices" = array<i32: 0, 0>}> : (!llvm.ptr) -> !llvm.ptr

  // test dso_local flag
  "llvm.mlir.global"() ({
  }) {"global_type" = i32, "sym_name" = "dso_global", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "dso_local"} : () -> ()

  // test thread_local_ flag
  "llvm.mlir.global"() ({
  }) {"global_type" = i32, "sym_name" = "tls_global", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "thread_local_"} : () -> ()

  // test section and alignment attributes
  "llvm.mlir.global"() ({
  }) {"global_type" = i64, "sym_name" = "section_global", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "section" = ".custom", "alignment" = 16 : i64} : () -> ()
}

// CHECK: builtin.module {
// CHECK-NEXT:   "llvm.mlir.global"() <{global_type = !llvm.array<13 x i8>, constant, sym_name = "str0", linkage = #llvm.linkage<"internal">, value = "Hello world!\n", addr_space = 0 : i32, unnamed_addr = 0 : i64}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   %0 = "llvm.mlir.addressof"() <{global_name = @str0}> : () -> !llvm.ptr
// CHECK-NEXT:   %1 = "llvm.getelementptr"(%0) <{elem_type = !llvm.array<13 x i8>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, 0>}> : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:   "llvm.mlir.global"() <{global_type = i32, sym_name = "dso_global", linkage = #llvm.linkage<"internal">, dso_local, addr_space = 0 : i32}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   "llvm.mlir.global"() <{global_type = i32, sym_name = "tls_global", linkage = #llvm.linkage<"internal">, thread_local_, addr_space = 0 : i32}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   "llvm.mlir.global"() <{global_type = i64, sym_name = "section_global", linkage = #llvm.linkage<"internal">, alignment = 16 : i64, addr_space = 0 : i32, section = ".custom"}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
