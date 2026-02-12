// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// This test checks that the llvm.func operation can be parse the following
// properties.

llvm.func @add(%arg0 : i32 {llvm.noundef}, %arg1 : i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic"} {
  llvm.return %arg0 : i32
}

// CHECK:       builtin.module {
// CHECK-NEXT:    llvm.func @add(%arg0 : i32 {llvm.noundef}, %arg1 : i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic"} {
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
