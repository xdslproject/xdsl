// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

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

// -----

// CHECK: Expected shape type.
"test.op"() {
  missing_spec = !emitc.array<>
}: ()->()

// -----

// CHECK: Expected 'x' in shape delimiter, got GREATER
"test.op"() {
  illegal_array_missing_x = !emitc.array<10>
}: ()->()

// -----

// CHECK: Expected shape type.
"test.op"() {
  illegal_array_missing_type = !emitc.array<10x>
}: ()->()

// -----

// CHECK: EmitC array dimensions must have non-negative size
"test.op"() {
  illegal_array_dynamic_shape = !emitc.array<10x?xi32>
}: ()->()

// -----

// CHECK: '>' expected
"test.op"() {
  illegal_array_unranked = !emitc.array<*xi32>
}: ()->()
