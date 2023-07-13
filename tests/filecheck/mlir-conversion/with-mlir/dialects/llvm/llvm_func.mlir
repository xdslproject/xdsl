// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "llvm.func"() ({
  ^bb0(%arg0: i64):
    %0 = "llvm.mlir.constant"() {"value" = 1 : i64} : () -> i64
    %1, %2 = "llvm.call_intrinsic"(%arg0, %0) {"intrin" = "my_intrin"} : (i64, i64) -> (i64, i64)
    "llvm.return"() : () -> ()
  }) {CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i64, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf", visibility_ = 0 : i64} : () -> ()

  %0 = "test.op"() : () -> i64
  llvm.call @printf(%0) : (i64) -> ()

  "llvm.func"() ({
  }) {CConv = #llvm.cconv<swiftcc>, function_type = !llvm.func<void (i64)>, linkage = #llvm.linkage<external>, sym_name = "nop", visibility_ = 1 : i64} : () -> ()

}) : () -> ()

// CHECK:      "llvm.func"() ({
// CHECK-NEXT: ^{{.*}}(%{{.*}}: i64):
// CHECK-NEXT:   %{{.*}} = "llvm.mlir.constant"() {"value" = 1 : i64} : () -> i64
// CHECK-NEXT:   %{{.*}}, %{{.*}} = "llvm.call_intrinsic"(%{{.*}}, %{{.*}}) {"intrin" = "my_intrin"} : (i64, i64) -> (i64, i64)

// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }) {"CConv" = #llvm.cconv<ccc>, "function_type" = !llvm.func<void (i64, ...)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "printf", "visibility_" = 0 : i64} : () -> ()

// CHECK:      "llvm.call"(%{{.*}}) {"callee" = @printf, "fastmathFlags" = #llvm.fastmath<none>} : (i64) -> ()

// CHECK:      "llvm.func"() ({
// CHECK-NEXT: }) {"CConv" = #llvm.cconv<swiftcc>, "function_type" = !llvm.func<void (i64)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "nop", "visibility_" = 1 : i64} : () -> ()
