// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

// Check that the right diagnostics are emitted when parsing EmitC types with invalid definition.

// CHECK: EmitC array shape must not be empty
"test.op"() {
  empty = !emitc.array<f32>
}: ()->()

// -----

// CHECK: invalid array element type 'memref<1xi32>'
"test.op"() {
  bad_type = !emitc.array<1xmemref<1xi32>>
}: ()->()

// -----

// CHECK: EmitC array element type cannot be another EmitC_ArrayType.
"test.op"() {
  nested = !emitc.array<2x!emitc.array<3xf32>>
}: ()->()
