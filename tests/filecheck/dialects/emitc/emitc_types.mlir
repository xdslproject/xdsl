// RUN: XDSL_ROUNDTRIP

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

// CHECK: bf16_1D = !emitc.array<1xbf16>
// CHECK-SAME: f16_1D = !emitc.array<1xf16>
// CHECK-SAME: f32_2D = !emitc.array<4x2xf32>
// CHECK-SAME: f64_1D = !emitc.array<5xf64>
// CHECK-SAME: i1_0D = !emitc.array<0xi1>
// CHECK-SAME: i1_3D = !emitc.array<3x4x5xi1>
// CHECK-SAME: i32_1D = !emitc.array<10xi32>
// CHECK-SAME: index_1D = !emitc.array<1xindex>
"test.op"() {
  bf16_1D = !emitc.array<1xbf16>,
  f16_1D = !emitc.array<1xf16>,
  f32_2D = !emitc.array<4x2xf32>,
  f64_1D = !emitc.array<5xf64>,
  i1_0D = !emitc.array<0xi1>,
  i1_3D = !emitc.array<3x4x5xi1>,
  i32_1D = !emitc.array<10xi32>,
  index_1D = !emitc.array<1xindex>
}: ()->()

//===----------------------------------------------------------------------===//
// LValueType
//===----------------------------------------------------------------------===//

// CHECK: bf16 = !emitc.lvalue<bf16>
// CHECK-SAME: f16 = !emitc.lvalue<f16>
// CHECK-SAME: f32 = !emitc.lvalue<f32>
// CHECK-SAME: f64 = !emitc.lvalue<f64>
// CHECK-SAME: i32 = !emitc.lvalue<i32>
// CHECK-SAME: index = !emitc.lvalue<index>
// CHECK-SAME: opaque_int = !emitc.lvalue<!emitc.opaque<"int">>
// CHECK-SAME: ptr_i32 = !emitc.lvalue<!emitc.ptr<i32>>
// CHECK-SAME: tensor_i32 = !emitc.lvalue<tensor<1xi32>>
// CHECK-SAME: tuple_i32 = !emitc.lvalue<tuple<i32, i32>>
"test.op"() {
  bf16 = !emitc.lvalue<bf16>,
  f16 = !emitc.lvalue<f16>,
  f32 = !emitc.lvalue<f32>,
  f64 = !emitc.lvalue<f64>,
  i32 = !emitc.lvalue<i32>,
  index = !emitc.lvalue<index>,
  opaque_int = !emitc.lvalue<!emitc.opaque<"int">>,
  ptr_i32 = !emitc.lvalue<!emitc.ptr<i32>>,
  tensor_i32 = !emitc.lvalue<tensor<1xi32>>,
  tuple_i32 = !emitc.lvalue<tuple<i32, i32>>
}: ()->()


//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

// CHECK: opaque_type = !emitc.opaque<"my_custom_type">
"test.op"() {
  opaque_type = !emitc.opaque<"my_custom_type">
}: ()->()

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// CHECK: bf16 = !emitc.ptr<bf16>
// CHECK-SAME: f16 = !emitc.ptr<f16>
// CHECK-SAME: f32 = !emitc.ptr<f32>
// CHECK-SAME: f64 = !emitc.ptr<f64>
// CHECK-SAME: i32 = !emitc.ptr<i32>
// CHECK-SAME: i64 = !emitc.ptr<i64>
// CHECK-SAME: ptr_i32 = !emitc.ptr<!emitc.ptr<i32>>
// CHECK-SAME: ptr_opaque_int = !emitc.ptr<!emitc.opaque<"int">>
"test.op"() {
  bf16 = !emitc.ptr<bf16>,
  f16 = !emitc.ptr<f16>,
  f32 = !emitc.ptr<f32>,
  f64 = !emitc.ptr<f64>,
  i32 = !emitc.ptr<i32>,
  i64 = !emitc.ptr<i64>,
  ptr_i32 = !emitc.ptr<!emitc.ptr<i32>>,
  ptr_opaque_int = !emitc.ptr<!emitc.opaque<"int">>
}: ()->()

//===----------------------------------------------------------------------===//
// PtrDiffTType
//===----------------------------------------------------------------------===//

// CHECK: array_ptrdiff = !emitc.array<30x!emitc.ptrdiff_t>
"test.op"() {
  array_ptrdiff = !emitc.array<30x!emitc.ptrdiff_t>
}: ()->()

//===----------------------------------------------------------------------===//
// SignedSizeTType
//===----------------------------------------------------------------------===//

// CHECK: array_ssize = !emitc.array<30x!emitc.ssize_t>
"test.op"() {
  array_ssize = !emitc.array<30x!emitc.ssize_t>
}: ()->()

//===----------------------------------------------------------------------===//
// SizeTType
//===----------------------------------------------------------------------===//

// CHECK: array_size = !emitc.array<30x!emitc.size_t>
"test.op"() {
  array_size = !emitc.array<30x!emitc.size_t>
}: ()->()
