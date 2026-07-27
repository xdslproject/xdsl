// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
  %i32 = "test.op"() : () -> i32
  %i64 = "test.op"() : () -> i64
  %f32 = "test.op"() : () -> f32
  %f64 = "test.op"() : () -> f64

  // CHECK: %i32_sum = wasmssa.add %i32 %i32 : i32
  %i32_sum = wasmssa.add %i32 %i32 : i32
  // CHECK: %i64_sum = wasmssa.add %i64 %i64 : i64
  %i64_sum = wasmssa.add %i64 %i64 : i64
  // CHECK: %f32_sum = wasmssa.add %f32 %f32 : f32
  %f32_sum = wasmssa.add %f32 %f32 : f32
  // CHECK: %f64_sum = wasmssa.add %f64 %f64 : f64
  %f64_sum = wasmssa.add %f64 %f64 : f64
}

// CHECK-GENERIC: "wasmssa.add"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.add"(%i64, %i64) : (i64, i64) -> i64
// CHECK-GENERIC: "wasmssa.add"(%f32, %f32) : (f32, f32) -> f32
// CHECK-GENERIC: "wasmssa.add"(%f64, %f64) : (f64, f64) -> f64
