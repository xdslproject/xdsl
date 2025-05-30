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
