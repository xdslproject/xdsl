// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<12 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "Hello world!", "unnamed_addr" = 0 : i64} : () -> ()
  "llvm.mlir.global"() ({
  }) {"global_type" = i32, "sym_name" = "data", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = 0 : i32, "unnamed_addr" = 0 : i64} : () -> ()
  %0 = "llvm.mlir.addressof"() {"global_name" = @data} : () -> !llvm.ptr
  %1 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "llvm.mlir.global"() <{addr_space = 0 : i32, constant, global_type = !llvm.array<12 x i8>, linkage = #llvm.linkage<"internal">, sym_name = "str0", unnamed_addr = 0 : i64, value = "Hello world!", visibility_ = 0 : i64}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   "llvm.mlir.global"() <{addr_space = 0 : i32, constant, global_type = i32, linkage = #llvm.linkage<"internal">, sym_name = "data", unnamed_addr = 0 : i64, value = 0 : i32, visibility_ = 0 : i64}> ({
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   %0 = "llvm.mlir.addressof"() <{global_name = @data}> : () -> !llvm.ptr
// CHECK-NEXT:   %1 = "llvm.mlir.zero"() : () -> !llvm.struct<(i32, f32)>
// CHECK-NEXT: }) : () -> ()
