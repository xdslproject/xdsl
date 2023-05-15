// RUN: xdsl-opt %s --print-op-generic | filecheck %s
"builtin.module"() ({
  %0 = "llvm.mlir.undef"() : () -> !llvm.array<2 x i64>
  %1 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i64>
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "llvm.mlir.undef"() : () -> !llvm.array<2 x i64>
// CHECK-NEXT:   %1 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i64>
// CHECK-NEXT: }) : () -> ()
