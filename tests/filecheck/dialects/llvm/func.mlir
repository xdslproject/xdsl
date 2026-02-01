// RUN: xdsl-opt %s | filecheck %s

// This test checks that the llvm.func operation can be parse the following
// properties.

"llvm.func"()
    <{
        CConv = #llvm.cconv<ccc>,
        arg_attrs = [{llvm.noundef}, {llvm.noundef}],
        frame_pointer = #llvm.framePointerKind<"non-leaf">,
        function_type = !llvm.func<i32 (i32, i32)>,
        linkage = #llvm.linkage<external>,
        no_inline,
        no_unwind,
        optimize_none,
        passthrough = [["no-trapping-math", "true"]],
        sym_name = "add",
        target_cpu = "x86-64",
        target_features = #llvm.target_features<["+mmx"]>,
        tune_cpu = "generic",
        unnamed_addr = 0 : i64,
        visibility_ = 0 : i64
    }> ({
  ^bb0(%arg0: i32, %arg1: i32):
    "llvm.return"(%arg0) : (i32) -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    llvm.func @add(%arg0 : i32 {llvm.noundef}, %arg1 : i32 {llvm.noundef}) -> i32 attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic", unnamed_addr = 0 : i64} {
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }

// Checks support for custom format

llvm.func @external_func(i64)

// CHECK: llvm.func @external_func(i64)

llvm.func @void_func(%arg0: i64) {
  llvm.return
}

// CHECK: llvm.func @void_func(%{{.*}} : i64) {
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

llvm.func @complex_func(%arg0: i64, %arg1: !llvm.ptr) -> i64 {
  %0 = llvm.add %arg0, %arg0 : i64
  llvm.return %0 : i64
}

// CHECK: llvm.func @complex_func(%{{.*}} : i64, %{{.*}} : !llvm.ptr) -> i64 {
// CHECK-NEXT:   %{{.*}} = llvm.add %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:   llvm.return %{{.*}} : i64
// CHECK-NEXT: }

llvm.func internal @internal_func() {
  llvm.return
}

// CHECK: llvm.func internal @internal_func() {
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

llvm.func @variadic_func(%arg0: i32, ...) {
  llvm.return
}

// CHECK: llvm.func @variadic_func(%{{.*}} : i32, ...) {
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

llvm.func @variadic_decl(i32, ...)

// CHECK: llvm.func @variadic_decl(i32, ...)

llvm.func @variadic_with_return(i32, ...) -> i64

// CHECK: llvm.func @variadic_with_return(i32, ...) -> i64

llvm.func @variadic_only(...)

// CHECK: llvm.func @variadic_only(...)
