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

// CHECK: Expected shape type.
"test.op"() {
  illegal_array_unranked = !emitc.array<*xi32>
}: ()->()

// -----

// CHECK: EmitC array element type '!emitc.lvalue<i32>' is not a supported EmitC type.
"test.op"() {
  lvalue_element_type = !emitc.array<4x!emitc.lvalue<i32>>
}: ()->()

// -----

// CHECK: !emitc.lvalue cannot wrap !emitc.array type
"test.op"() {
  illegal_lvalue_type_1 = !emitc.lvalue<!emitc.array<1xi32>>
}: ()->()

// -----

// CHECK: !emitc.lvalue must wrap supported emitc type, but got !emitc.lvalue<i32>
"test.op"() {
  illegal_lvalue_type_2 = !emitc.lvalue<!emitc.lvalue<i32>>
}: ()->()

// -----

// CHECK: !emitc.lvalue must wrap supported emitc type, but got i17
"test.op"() {
  illegal_lvalue_type_3 = !emitc.lvalue<i17>
}: ()->()

// -----

// CHECK: !emitc.lvalue must wrap supported emitc type, but got tensor<1x!emitc.array<1xi32>>
"test.op"() {
  illegal_lvalue_tensor_emitc_array_i32 = !emitc.lvalue<tensor<1x!emitc.array<1xi32>>>
}: ()->()

// -----

// CHECK: !emitc.lvalue must wrap supported emitc type, but got tuple<!emitc.array<1xi32>>
"test.op"() {
  illegal_lvalue_tuple_emitc_array_i32 = !emitc.lvalue<tuple<!emitc.array<1xi32>>>
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
