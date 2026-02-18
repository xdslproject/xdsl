// RUN: XDSL_ROUNDTRIP

llvm.func @add(%arg0 : i32 {llvm.noundef}, %arg1 : i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic"} {
  llvm.return %arg0 : i32
}

// CHECK:       builtin.module {
// CHECK-NEXT:    llvm.func @add(%arg0 : i32 {llvm.noundef}, %arg1 : i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic"} {
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }

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

llvm.func @func_with_attrs() attributes {hello = "world"} {
  llvm.return
}

// CHECK: llvm.func @func_with_attrs() attributes {hello = "world"} {
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

llvm.func private @wrapped_function(%arg0: i32, %arg1: i32) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
   "llvm.call"(%arg0, %arg1) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @_mlir_ciface_wrapped_function, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (i32, i32) -> ()
   //llvm.call @_mlir_ciface_wrapped_function(%arg0, %arg1) : (i32, i32) -> ()
   llvm.return
}

llvm.func @_mlir_ciface_wrapped_function(i32, i32) attributes {llvm.emit_c_interface, sym_visibility = "private"}

// CHECK:  llvm.func private @wrapped_function(%arg0_4 : i32, %arg1_2 : i32) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
// CHECK-NEXT:    "llvm.call"(%arg0_4, %arg1_2) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @_mlir_ciface_wrapped_function, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (i32, i32) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @_mlir_ciface_wrapped_function(i32, i32) attributes {llvm.emit_c_interface, sym_visibility = "private"}
