// RUN: xdsl-opt -t llvm --split-input-file %s | filecheck %s

// CHECK: ; ModuleID = ""
// CHECK-NEXT: target triple = "unknown-unknown-unknown"
// CHECK-NEXT: target datalayout = ""

// covers: FuncOp, FAddOp, ReturnOp (value), Type conversion (f64)
"llvm.func"() <{"sym_name" = "add", "function_type" = !llvm.func<f64 (f64, f64)>, "linkage" = #llvm.linkage<"external">, "CConv" = #llvm.cconv<ccc>, "visibility_" = 0 : i64}> ({
^bb0(%arg0: f64, %arg1: f64):
  %0 = "llvm.fadd"(%arg0, %arg1) <{"fastmathFlags" = #llvm.fastmath<none>}> : (f64, f64) -> f64
  "llvm.return"(%0) : (f64) -> ()
}) : () -> ()

// CHECK: define double @"add"(double %".1", double %".2")
// CHECK-NEXT: {
// CHECK-NEXT: block_0:
// CHECK-NEXT:   %"res" = fadd double %".1", %".2"
// CHECK-NEXT:   ret double %"res"
// CHECK-NEXT: }

// covers: ReturnOp (void), FunctionType (void return)
"llvm.func"() <{"sym_name" = "void_func", "function_type" = !llvm.func<void ()>, "linkage" = #llvm.linkage<"external">, "CConv" = #llvm.cconv<ccc>, "visibility_" = 0 : i64}> ({
^bb0:
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: define void @"void_func"()
// CHECK-NEXT: {
// CHECK-NEXT: block_0:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// covers: argument mapping loop
"llvm.func"() <{"sym_name" = "multi_arg", "function_type" = !llvm.func<f64 (f64, f64, f64)>, "linkage" = #llvm.linkage<"external">, "CConv" = #llvm.cconv<ccc>, "visibility_" = 0 : i64}> ({
^bb0(%a: f64, %b: f64, %c: f64):
  %0 = "llvm.fadd"(%a, %b) <{"fastmathFlags" = #llvm.fastmath<none>}> : (f64, f64) -> f64
  %1 = "llvm.fadd"(%0, %c) <{"fastmathFlags" = #llvm.fastmath<none>}> : (f64, f64) -> f64
  "llvm.return"(%1) : (f64) -> ()
}) : () -> ()

// CHECK: define double @"multi_arg"(double %".1", double %".2", double %".3")
// CHECK-NEXT: {
// CHECK-NEXT: block_0:
// CHECK-NEXT:   %"res" = fadd double %".1", %".2"
// CHECK-NEXT:   %"res.1" = fadd double %"res", %".3"
// CHECK-NEXT:   ret double %"res.1"
// CHECK-NEXT: }

// covers: block iteration
"llvm.func"() <{"sym_name" = "multi_block", "function_type" = !llvm.func<void ()>, "linkage" = #llvm.linkage<"external">, "CConv" = #llvm.cconv<ccc>, "visibility_" = 0 : i64}> ({
^bb0:
  "llvm.return"() : () -> ()
^bb1:
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: define void @"multi_block"()
// CHECK-NEXT: {
// CHECK-NEXT: block_0:
// CHECK-NEXT:   ret void
// CHECK-NEXT: block_1:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
