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

// -----

// CHECK: !emitc.lvalue must wrap supported emitc type, but got memref<1xi32>
"test.op"() {
  unsupported = !emitc.lvalue<memref<1xi32>>
}: ()->()

// -----

// CHECK: !emitc.lvalue cannot wrap !emitc.array type
"test.op"() {
  wrap_array = !emitc.lvalue<!emitc.array<1xf32>>
}: ()->()

// -----

// CHECK: !emitc.lvalue must wrap supported emitc type, but got !emitc.lvalue<i32>
"test.op"() {
  nested_lvalue = !emitc.lvalue<!emitc.lvalue<i32>>
}: ()->()

// -----

// CHECK: expected non empty string in !emitc.opaque type
"test.op"() {
  empty_str = !emitc.opaque<"">
}: ()->()

// -----

// CHECK: pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead
"test.op"() {
  with_ptr = !emitc.opaque<"foo*">
}: ()->()

// -----

// CHECK: pointers to lvalues are not allowed
"test.op"() {
  ptr_lvalue = !emitc.ptr<!emitc.lvalue<i32>>
}: ()->()
