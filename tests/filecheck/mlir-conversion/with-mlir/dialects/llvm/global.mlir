// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt -f mlir -t mlir | filecheck %s
"builtin.module"() ({
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<12 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "Hello world!", "unnamed_addr" = 0 : i64} : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:  "llvm.mlir.global"() ({
// CHECK-NEXT:  }) {"addr_space" = 0 : i32, "constant", "global_type" = !llvm.array<12 x i8>, "linkage" = #llvm.linkage<"internal">, "sym_name" = "str0", "unnamed_addr" = 0 : i64, "value" = "Hello world!"} : () -> ()
// CHECK-NEXT: }) : () -> ()
