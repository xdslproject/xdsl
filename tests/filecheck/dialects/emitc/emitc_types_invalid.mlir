// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

// Check that the right diagnostics are emitted when verify EmitC types with invalid definition.

// CHECK: EmitC array shape must not be empty
"test.op"() {
  empty = !emitc.array<f32>
}: ()->()

// -----

// CHECK: Invalid value 0, expected one of {32, 1, 64, 16, 8}
"test.op"() {
  bad_type = !emitc.array<1xi0>>
}: ()->()

// -----

// CHECK: Unexpected attribute memref<1xi32>
"test.op"() {
  bad_type = !emitc.array<1xmemref<1xi32>>
}: ()->()

// -----

// CHECK: Unexpected attribute !emitc.array<3xf32>
"test.op"() {
  nested = !emitc.array<2x!emitc.array<3xf32>>
}: ()->()

// -----

// CHECK: Unexpected attribute tensor<1x!emitc.array<1xf32>>
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

// CHECK: expected static shape, but got dynamic dimension
"test.op"() {
  illegal_array_dynamic_shape = !emitc.array<10x?xi32>
}: ()->()

// -----

// CHECK: Expected shape type.
"test.op"() {
  illegal_array_unranked = !emitc.array<*xi32>
}: ()->()

// -----

// CHECK: Unexpected attribute !emitc.lvalue<i32>
"test.op"() {
  lvalue_element_type = !emitc.array<4x!emitc.lvalue<i32>>
}: ()->()

// -----

// CHECK: !emitc.lvalue cannot wrap !emitc.array type
"test.op"() {
  illegal_lvalue_type_1 = !emitc.lvalue<!emitc.array<1xi32>>
}: ()->()

// -----

// CHECK: Unexpected attribute !emitc.lvalue<i32>
"test.op"() {
  illegal_lvalue_type_2 = !emitc.lvalue<!emitc.lvalue<i32>>
}: ()->()

// -----

// CHECK: Invalid value 17, expected one of {32, 1, 64, 16, 8}
"test.op"() {
  illegal_lvalue_type_3 = !emitc.lvalue<i17>
}: ()->()

// -----

// CHECK: EmitC type cannot be a tensor of EmitC arrays
"test.op"() {
  illegal_lvalue_tensor_emitc_array_i32 = !emitc.lvalue<tensor<1x!emitc.array<1xi32>>>
}: ()->()

// -----

// CHECK: EmitC type cannot be a tuple of EmitC arrays
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

// -----

// CHECK: pointers to lvalues are not allowed
"test.op"() {
  ptr_lvalue = !emitc.ptr<!emitc.lvalue<i32>>
}: ()->()

// -----

// CHECK: Unexpected attribute memref<1xi32>
"test.op"() {
  lvalue_ptr_memref = !emitc.lvalue<!emitc.ptr<memref<1xi32>>>
}: ()->()

// -----

// CHECK: Unexpected attribute f80
"test.op"() {
  unsupported_f80 = !emitc.array<1xf80>
}: ()->()

// -----

// CHECK: Unexpected attribute f80
"test.op"() {
  unsupported_f80_lvalue = !emitc.lvalue<f80>
}: ()->()
