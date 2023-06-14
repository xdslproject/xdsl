// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "llvm.func"() ({
  }) {CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i64, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf", visibility_ = 0 : i64} : () -> ()

  %0 = "test.op"() : () -> i64
  llvm.call @printf(%0) : (i64) -> ()

  "llvm.func"() ({
  }) {CConv = #llvm.cconv<swiftcc>, function_type = !llvm.func<void (i64)>, linkage = #llvm.linkage<external>, sym_name = "nop", visibility_ = 1 : i64} : () -> ()

}) : () -> ()

// CHECK:      "llvm.func"() ({
// CHECK-NEXT: }) {"CConv" = #llvm.cconv<ccc>, "function_type" = !llvm.func<void (i64, ...)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "printf", "visibility_" = 0 : i64} : () -> ()

// CHECK:      "llvm.call"(%0) {"callee" = @printf, "fastmathFlags" = #llvm.fastmath<none>} : (i64) -> ()

// CHECK:      "llvm.func"() ({
// CHECK-NEXT: }) {"CConv" = #llvm.cconv<swiftcc>, "function_type" = !llvm.func<void (i64)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "nop", "visibility_" = 1 : i64} : () -> ()
