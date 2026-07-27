// RUN: MLIR_ROUNDTRIP

builtin.module {
  // CHECK: %{{.*}} = wasmssa.const 1 : i32
  %i32 = wasmssa.const 1 : i32
  // CHECK-NEXT: %{{.*}} = wasmssa.const 2 : i64
  %i64 = wasmssa.const 2 : i64
  // CHECK-NEXT: %{{.*}} = wasmssa.const 3.000000e+00 : f32
  %f32 = wasmssa.const 3.000000e+00 : f32
  // CHECK-NEXT: %{{.*}} = wasmssa.const 4.000000e+00 : f64
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
}
