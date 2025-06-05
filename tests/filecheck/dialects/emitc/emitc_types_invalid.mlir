// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

// Check that the right diagnostics are emitted when verify EmitC types with invalid definition.

// CHECK: EmitC array shape must not be empty
"test.op"() {
  empty = !emitc.array<f32>
}: ()->()

// -----

// CHECK: EmitC array element type 'i0' is not a supported EmitC type.
"test.op"() {
  bad_type = !emitc.array<1xi0>>
}: ()->()

// -----

// CHECK: EmitC array element type 'memref<1xi32>' is not a supported EmitC type.
"test.op"() {
  bad_type = !emitc.array<1xmemref<1xi32>>
}: ()->()

// -----

// CHECK: EmitC array element type cannot be another EmitC_ArrayType.
"test.op"() {
  nested = !emitc.array<2x!emitc.array<3xf32>>
}: ()->()

// -----

// CHECK: EmitC array element type 'tensor<1x!emitc.array<1xf32>>' is not a supported EmitC type.
"test.op"() {
  tensor_with_emitc_array = !emitc.array<1xtensor<1x!emitc.array<1xf32>>>
}: ()->()
