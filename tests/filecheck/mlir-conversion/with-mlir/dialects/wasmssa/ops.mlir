// RUN: MLIR_ROUNDTRIP

// CHECK: %[[I32:.*]] = wasmssa.const 1 : i32
%i32 = wasmssa.const 1 : i32
// CHECK-NEXT: %[[I64:.*]] = wasmssa.const 2 : i64
%i64 = wasmssa.const 2 : i64
// CHECK-NEXT: %[[F32:.*]] = wasmssa.const 3.000000e+00 : f32
%f32 = wasmssa.const 3.000000e+00 : f32
// CHECK-NEXT: %[[F64:.*]] = wasmssa.const 4.000000e+00 : f64
%f64 = wasmssa.const 4.000000e+00 : f64

// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_i32 : i32
%global_i32 = wasmssa.global_get @global_i32 : i32
// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_i64 : i64
%global_i64 = wasmssa.global_get @global_i64 : i64
// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_i128 : i128
%global_i128 = wasmssa.global_get @global_i128 : i128
// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_f32 : f32
%global_f32 = wasmssa.global_get @global_f32 : f32
// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_f64 : f64
%global_f64 = wasmssa.global_get @global_f64 : f64
// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_funcref : !wasmssa.funcref
%global_funcref = wasmssa.global_get @global_funcref : !wasmssa.funcref
// CHECK-NEXT: %{{.*}} = wasmssa.global_get @global_externref : !wasmssa.externref
%global_externref = wasmssa.global_get @global_externref : !wasmssa.externref

// CHECK-NEXT: %{{.*}} = wasmssa.add %[[I32]] %[[I32]] : i32
%i32_sum = wasmssa.add %i32 %i32 : i32
// CHECK-NEXT: %{{.*}} = wasmssa.add %[[I64]] %[[I64]] : i64
%i64_sum = wasmssa.add %i64 %i64 : i64
// CHECK-NEXT: %{{.*}} = wasmssa.add %[[F32]] %[[F32]] : f32
%f32_sum = wasmssa.add %f32 %f32 : f32
// CHECK-NEXT: %{{.*}} = wasmssa.add %[[F64]] %[[F64]] : f64
%f64_sum = wasmssa.add %f64 %f64 : f64
