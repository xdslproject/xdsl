// RUN: MLIR_ROUNDTRIP

builtin.module {
  // CHECK: %[[I32:.*]] = "test.op"() : () -> i32
  %i32 = "test.op"() : () -> i32
  // CHECK: %[[I64:.*]] = "test.op"() : () -> i64
  %i64 = "test.op"() : () -> i64
  // CHECK: %[[F32:.*]] = "test.op"() : () -> f32
  %f32 = "test.op"() : () -> f32
  // CHECK: %[[F64:.*]] = "test.op"() : () -> f64
  %f64 = "test.op"() : () -> f64

  // CHECK: %{{.*}} = wasmssa.add %[[I32]] %[[I32]] : i32
  %i32_sum = wasmssa.add %i32 %i32 : i32
  // CHECK: %{{.*}} = wasmssa.add %[[I64]] %[[I64]] : i64
  %i64_sum = wasmssa.add %i64 %i64 : i64
  // CHECK: %{{.*}} = wasmssa.add %[[F32]] %[[F32]] : f32
  %f32_sum = wasmssa.add %f32 %f32 : f32
  // CHECK: %{{.*}} = wasmssa.add %[[F64]] %[[F64]] : f64
  %f64_sum = wasmssa.add %f64 %f64 : f64
}
