// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "llvm.func"() ({
  }) {CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i32, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf", visibility_ = 0 : i64} : () -> ()
}) : () -> ()

// CHECK:      "llvm.func"() ({
// CHECK-NEXT: }) {"CConv" = #llvm.cconv<ccc>, "function_type" = !llvm.func<void (i32, ...)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "printf", "visibility_" = 0 : i64} : () -> ()
