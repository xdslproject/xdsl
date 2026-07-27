// RUN: MLIR_ROUNDTRIP

// CHECK: %[[I32:.*]] = "test.op"() : () -> i32
%i32 = "test.op"() : () -> i32
// CHECK-NEXT: %[[I64:.*]] = "test.op"() : () -> i64
%i64 = "test.op"() : () -> i64
// CHECK-NEXT: %[[F32:.*]] = "test.op"() : () -> f32
%f32 = "test.op"() : () -> f32
// CHECK-NEXT: %[[F64:.*]] = "test.op"() : () -> f64
%f64 = "test.op"() : () -> f64

// CHECK-NEXT: %{{.*}} = wasmssa.add %[[I32]] %[[I32]] : i32
%i32_sum = wasmssa.add %i32 %i32 : i32
// CHECK-NEXT: %{{.*}} = wasmssa.add %[[I64]] %[[I64]] : i64
%i64_sum = wasmssa.add %i64 %i64 : i64
// CHECK-NEXT: %{{.*}} = wasmssa.add %[[F32]] %[[F32]] : f32
%f32_sum = wasmssa.add %f32 %f32 : f32
// CHECK-NEXT: %{{.*}} = wasmssa.add %[[F64]] %[[F64]] : f64
%f64_sum = wasmssa.add %f64 %f64 : f64
