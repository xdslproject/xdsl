// RUN: mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %cst = arith.constant {"truc" = !llvm.func<ptr<16> (i64, ...)>} 10 : i32
  "llvm.func"() ({
  ^bb0(%arg0: i64):
    %0 = "llvm.mlir.constant"() {"value" = 1 : i64} : () -> i64
    %1 = "llvm.call_intrinsic"(%arg0, %0) <{intrin = "llvm.my_intrin", op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (i64, i64) -> (i64)
    "llvm.return"() : () -> ()
  }) {CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i64, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf", visibility_ = 0 : i64} : () -> ()

  %0 = "test.op"() : () -> i64
  llvm.call @printf(%0) vararg(!llvm.func<void (i64, ...)>) : (i64) -> ()

  "llvm.func"() ({
  }) {CConv = #llvm.cconv<swiftcc>, function_type = !llvm.func<void (i64)>, linkage = #llvm.linkage<external>, sym_name = "nop", visibility_ = 1 : i64} : () -> ()

}) : () -> ()

// CHECK:      %{{.*}} = "arith.constant"() <{value = 10 : i32}> {truc = !llvm.func<!llvm.ptr<16> (i64, ...)>} : () -> i32
// CHECK-NEXT: "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i64, ...)>, linkage = #llvm.linkage<"external">, sym_name = "printf", visibility_ = 0 : i64}> ({
// CHECK-NEXT: ^{{.*}}(%arg0 : i64):
// CHECK-NEXT:   %{{.*}} = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %{{.*}} = "llvm.call_intrinsic"(%arg0, %{{.*}}) <{fastmathFlags = #llvm.fastmath<none>, intrin = "llvm.my_intrin", op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (i64, i64) -> i64
// CHECK-NEXT:   "llvm.return"() : () -> ()
// CHECK-NEXT: }) : () -> ()
// CHECK-NEXT: %{{.*}} = "test.op"() : () -> i64
// CHECK-NEXT: "llvm.call"(%{{.*}}) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @printf, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, var_callee_type = !llvm.func<void (i64, ...)>}> : (i64) -> ()
// CHECK-NEXT: "llvm.func"() <{CConv = #llvm.cconv<swiftcc>, function_type = !llvm.func<void (i64)>, linkage = #llvm.linkage<"external">, sym_name = "nop", visibility_ = 1 : i64}> ({
// CHECK-NEXT: }) : () -> ()
