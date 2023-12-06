// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %cst = arith.constant {"truc" = !llvm.func<ptr<16> (i64, ...)>} 10 : i32
  "llvm.func"() ({
  ^bb0(%arg0: i64):
    %0 = "llvm.mlir.constant"() {"value" = 1 : i64} : () -> i64
    %1 = "llvm.call_intrinsic"(%arg0, %0) {"intrin" = "llvm.my_intrin"} : (i64, i64) -> (i64)
    "llvm.return"() : () -> ()
  }) {CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i64, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf", visibility_ = 0 : i64} : () -> ()

  %0 = "test.op"() : () -> i64
  llvm.call @printf(%0) vararg(!llvm.func<i32 (ptr, ...)>) : (i64) -> ()

  "llvm.func"() ({
  }) {CConv = #llvm.cconv<swiftcc>, function_type = !llvm.func<void (i64)>, linkage = #llvm.linkage<external>, sym_name = "nop", visibility_ = 1 : i64} : () -> ()

}) : () -> ()

// CHECK:      "llvm.func"() <{"CConv" = #llvm.cconv<ccc>, "function_type" = !llvm.func<void (i64, ...)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "printf", "visibility_" = 0 : i64}> ({
// CHECK-NEXT: ^{{.*}}(%{{.*}}: i64):
// CHECK-NEXT:   %{{.*}} = "llvm.mlir.constant"() <{"value" = 1 : i64}> : () -> i64
// CHECK-NEXT:   %{{.*}} = "llvm.call_intrinsic"(%{{.*}}, %{{.*}}) <{"fastmathFlags" = #llvm.fastmath<none>, "intrin" = "llvm.my_intrin"}> : (i64, i64) -> i64

// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK:      "llvm.call"(%{{.*}}) <{"CConv" = #llvm.cconv<ccc>, "callee" = @printf, "callee_type" = !llvm.func<i32 (!llvm.ptr, ...)>, "fastmathFlags" = #llvm.fastmath<none>}> : (i64) -> ()

// CHECK:      "llvm.func"() <{"CConv" = #llvm.cconv<swiftcc>, "function_type" = !llvm.func<void (i64)>, "linkage" = #llvm.linkage<"external">, "sym_name" = "nop", "visibility_" = 1 : i64}> ({
// CHECK-NEXT: }) : () -> ()
