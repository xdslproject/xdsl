// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

// Check that the right diagnostics are emitted when verify EmitC types with invalid definition.

// CHECK: EmitC array shape must not be empty
"test.op"() {
  empty = !emitc.array<f32>
}: ()->()
