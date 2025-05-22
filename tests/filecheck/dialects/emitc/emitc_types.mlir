// RUN: XDSL_ROUNDTRIP

// CHECK: f32_2D = !emitc.array<4x2xf32>
// CHECK-SAME: i32_1D = !emitc.array<10xi32>
// CHECK-SAME: f64_1D = !emitc.array<5xf64>
// CHECK-SAME: i1_3D = !emitc.array<3x4x5xi1>
// CHECK-SAME: i1_0D = !emitc.array<0xi1>
// CHECK-SAME: lvalue_i32 = !emitc.lvalue<i32>
// CHECK-SAME: opaque_type = !emitc.opaque<"my_custom_type">
// CHECK-SAME: ptr_f32 = !emitc.ptr<f32>
// CHECK-SAME: ptrdiff_t_type = !emitc.ptrdiff_t
// CHECK-SAME: size_t_type = !emitc.size_t
// CHECK-SAME: ssize_t_type = !emitc.ssize_t
// CHECK-SAME: opaque_attr = #emitc.opaque<"my_custom_attr">
"test.op"() {
  f32_2D = !emitc.array<4x2xf32>,
  i32_1D = !emitc.array<10xi32>,
  f64_1D = !emitc.array<5xf64>,
  i1_3D = !emitc.array<3x4x5xi1>,
  i1_0D = !emitc.array<0xi1>,
  lvalue_i32 = !emitc.lvalue<i32>,
  opaque_type = !emitc.opaque<"my_custom_type">,
  ptr_f32 = !emitc.ptr<f32>,
  ptrdiff_t_type = !emitc.ptrdiff_t,
  size_t_type = !emitc.size_t,
  ssize_t_type = !emitc.ssize_t,
  opaque_attr = #emitc.opaque<"my_custom_attr">
}: ()->()
