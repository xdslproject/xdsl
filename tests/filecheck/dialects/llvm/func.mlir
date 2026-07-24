// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt %s --print-debuginfo | filecheck %s --check-prefix=CHECK-DEBUG-INFO

llvm.func @add(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic"} {
  llvm.return %arg0 : i32
}

// CHECK:       builtin.module {
// CHECK-NEXT:    llvm.func @add(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic"} {
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }

llvm.func @external_func(i64)

// CHECK: llvm.func @external_func(i64)

llvm.func @unnamed_arg_attrs_loc(i64 {llvm.noundef} loc("model.mlir":7:9))

// CHECK: llvm.func @unnamed_arg_attrs_loc(i64)
// CHECK-DEBUG-INFO: llvm.func @unnamed_arg_attrs_loc(i64)

llvm.func @named_arg_attrs_loc(%arg0: i64 {llvm.noundef} loc("model.mlir":8:11)) {
  llvm.return
}

// CHECK: llvm.func @named_arg_attrs_loc(%{{.*}}: i64 {llvm.noundef}) {
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }
// CHECK-DEBUG-INFO: llvm.func @named_arg_attrs_loc(%{{.*}}: i64 {llvm.noundef} loc("model.mlir":8:11)) {
// CHECK-DEBUG-INFO-NEXT:   llvm.return
// CHECK-DEBUG-INFO-NEXT: }

llvm.func @void_func(%arg0: i64) {
  llvm.return
}

// CHECK: llvm.func @void_func(%{{.*}}: i64) {
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

llvm.func @complex_func(%arg0: i64, %arg1: !llvm.ptr) -> i64 {
  %0 = llvm.add %arg0, %arg0 : i64
  llvm.return %0 : i64
}

// CHECK: llvm.func @complex_func(%{{.*}}: i64, %{{.*}}: !llvm.ptr) -> i64 {
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

// CHECK: llvm.func @variadic_func(%{{.*}}: i32, ...) {
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
   llvm.call @_mlir_ciface_wrapped_function(%arg0, %arg1) : (i32, i32) -> ()
   llvm.return
}

llvm.func @_mlir_ciface_wrapped_function(i32, i32) attributes {llvm.emit_c_interface, sym_visibility = "private"}

// CHECK:  llvm.func private @wrapped_function(%{{.*}}: i32, %{{.*}}: i32) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
// CHECK-NEXT:    llvm.call @_mlir_ciface_wrapped_function(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @_mlir_ciface_wrapped_function(i32, i32) attributes {llvm.emit_c_interface, sym_visibility = "private"}

llvm.func @float_callee(f32) -> f32
llvm.func @variadic_callee(i32, ...) -> i32

llvm.func @test_calls(%arg0: i32, %fptr: !llvm.ptr, %farg: f32) {
  llvm.call @_mlir_ciface_wrapped_function(%arg0, %arg0) : (i32, i32) -> ()
  %0 = llvm.call %fptr(%arg0) : !llvm.ptr, (i32) -> i32
  llvm.call tail @external_func(%arg0) : (i32) -> ()
  llvm.call fastcc @external_func(%arg0) : (i32) -> ()
  %1 = llvm.call @variadic_callee(%arg0) vararg(!llvm.func<i32 (i32, ...)>) : (i32) -> i32
  %2 = llvm.call @float_callee(%farg) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  llvm.return
}

// CHECK: llvm.func @float_callee(f32) -> f32
// CHECK-NEXT: llvm.func @variadic_callee(i32, ...) -> i32
// CHECK-NEXT: llvm.func @test_calls(%{{.*}}: i32, %{{.*}}: !llvm.ptr, %{{.*}}: f32) {
// CHECK-NEXT:   llvm.call @_mlir_ciface_wrapped_function(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
// CHECK-NEXT:   %{{.*}} = llvm.call %{{.*}}(%{{.*}}) : !llvm.ptr, (i32) -> i32
// CHECK-NEXT:   llvm.call tail @external_func(%{{.*}}) : (i32) -> ()
// CHECK-NEXT:   llvm.call fastcc @external_func(%{{.*}}) : (i32) -> ()
// CHECK-NEXT:   %{{.*}} = llvm.call @variadic_callee(%{{.*}}) vararg(!llvm.func<i32 (i32, ...)>) : (i32) -> i32
// CHECK-NEXT:   %{{.*}} = llvm.call @float_callee(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

%ci0, %ci1 = "test.op"() : () -> (i64, i64)
%ci2 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) : (i64, i64) -> i64
llvm.call_intrinsic "llvm.donothing"() : () -> ()
%ci3 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) {fastmathFlags = #llvm.fastmath<reassoc,nnan>} : (i64, i64) -> i64
%ci4 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) {fastmathFlags = #llvm.fastmath<fast>} : (i64, i64) -> i64

%ci5 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) [%ci0, %ci1 : i64, i64] {op_bundle_sizes = array<i32: 2>} : (i64, i64) -> i64

// CHECK: %ci0, %ci1 = "test.op"() : () -> (i64, i64)
// CHECK-NEXT: %ci2 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) : (i64, i64) -> i64
// CHECK-NEXT: llvm.call_intrinsic "llvm.donothing"() : () -> ()
// CHECK-NEXT: %ci3 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) {fastmathFlags = #llvm.fastmath<reassoc,nnan>} : (i64, i64) -> i64
// CHECK-NEXT: %ci4 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) {fastmathFlags = #llvm.fastmath<fast>} : (i64, i64) -> i64
// CHECK-NEXT: %ci5 = llvm.call_intrinsic "llvm.smax"(%ci0, %ci1) [%ci0, %ci1 : i64, i64] {op_bundle_sizes = array<i32: 2>} : (i64, i64) -> i64
