// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: %i32 = wasmssa.const 1 : i32
%i32 = wasmssa.const 1 : i32
// CHECK-NEXT: %i64 = wasmssa.const 2 : i64
%i64 = wasmssa.const 2 : i64
// CHECK-NEXT: %f32 = wasmssa.const 3.000000e+00 : f32
%f32 = wasmssa.const 3.000000e+00 : f32
// CHECK-NEXT: %f64 = wasmssa.const 4.000000e+00 : f64
%f64 = wasmssa.const 4.000000e+00 : f64

// CHECK-NEXT: %global_i32 = wasmssa.global_get @global_i32 : i32
%global_i32 = wasmssa.global_get @global_i32 : i32
// CHECK-NEXT: %global_i64 = wasmssa.global_get @global_i64 : i64
%global_i64 = wasmssa.global_get @global_i64 : i64
// CHECK-NEXT: %global_i128 = wasmssa.global_get @global_i128 : i128
%global_i128 = wasmssa.global_get @global_i128 : i128
// CHECK-NEXT: %global_f32 = wasmssa.global_get @global_f32 : f32
%global_f32 = wasmssa.global_get @global_f32 : f32
// CHECK-NEXT: %global_f64 = wasmssa.global_get @global_f64 : f64
%global_f64 = wasmssa.global_get @global_f64 : f64
// CHECK-NEXT: %global_funcref = wasmssa.global_get @global_funcref : !wasmssa.funcref
%global_funcref = wasmssa.global_get @global_funcref : !wasmssa.funcref
// CHECK-NEXT: %global_externref = wasmssa.global_get @global_externref : !wasmssa.externref
%global_externref = wasmssa.global_get @global_externref : !wasmssa.externref

// CHECK-NEXT: %i32_sum = wasmssa.add %i32 %i32 : i32
%i32_sum = wasmssa.add %i32 %i32 : i32
// CHECK-NEXT: %i64_sum = wasmssa.add %i64 %i64 : i64
%i64_sum = wasmssa.add %i64 %i64 : i64
// CHECK-NEXT: %f32_sum = wasmssa.add %f32 %f32 : f32
%f32_sum = wasmssa.add %f32 %f32 : f32
// CHECK-NEXT: %f64_sum = wasmssa.add %f64 %f64 : f64
%f64_sum = wasmssa.add %f64 %f64 : f64
// CHECK-NEXT: %i32_and = wasmssa.and %i32 %i32 : i32
%i32_and = wasmssa.and %i32 %i32 : i32
// CHECK-NEXT: %f32_div = wasmssa.div %f32 %f32 : f32
%f32_div = wasmssa.div %f32 %f32 : f32
// CHECK-NEXT: %i32_div_ui = wasmssa.div_ui %i32 %i32 : i32
%i32_div_ui = wasmssa.div_ui %i32 %i32 : i32
// CHECK-NEXT: %i32_div_si = wasmssa.div_si %i32 %i32 : i32
%i32_div_si = wasmssa.div_si %i32 %i32 : i32
// CHECK-NEXT: %i32_mul = wasmssa.mul %i32 %i32 : i32
%i32_mul = wasmssa.mul %i32 %i32 : i32
// CHECK-NEXT: %i32_or = wasmssa.or %i32 %i32 : i32
%i32_or = wasmssa.or %i32 %i32 : i32
// CHECK-NEXT: %i32_sub = wasmssa.sub %i32 %i32 : i32
%i32_sub = wasmssa.sub %i32 %i32 : i32
// CHECK-NEXT: %i32_rem_ui = wasmssa.rem_ui %i32 %i32 : i32
%i32_rem_ui = wasmssa.rem_ui %i32 %i32 : i32
// CHECK-NEXT: %i32_rem_si = wasmssa.rem_si %i32 %i32 : i32
%i32_rem_si = wasmssa.rem_si %i32 %i32 : i32
// CHECK-NEXT: %i32_xor = wasmssa.xor %i32 %i32 : i32
%i32_xor = wasmssa.xor %i32 %i32 : i32
// CHECK-NEXT: %f32_min = wasmssa.min %f32 %f32 : f32
%f32_min = wasmssa.min %f32 %f32 : f32
// CHECK-NEXT: %f32_max = wasmssa.max %f32 %f32 : f32
%f32_max = wasmssa.max %f32 %f32 : f32
// CHECK-NEXT: %f32_copysign = wasmssa.copysign %f32 %f32 : f32
%f32_copysign = wasmssa.copysign %f32 %f32 : f32

// CHECK-GENERIC: "wasmssa.const"() <{value = 1 : i32}> : () -> i32
// CHECK-GENERIC: "wasmssa.const"() <{value = 2 : i64}> : () -> i64
// CHECK-GENERIC: "wasmssa.const"() <{value = 3.000000e+00 : f32}> : () -> f32
// CHECK-GENERIC: "wasmssa.const"() <{value = 4.000000e+00 : f64}> : () -> f64
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_i32}> : () -> i32
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_i64}> : () -> i64
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_i128}> : () -> i128
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_f32}> : () -> f32
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_f64}> : () -> f64
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_funcref}> : () -> !wasmssa.funcref
// CHECK-GENERIC: "wasmssa.global_get"() <{global = @global_externref}> : () -> !wasmssa.externref
// CHECK-GENERIC: "wasmssa.add"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.add"(%i64, %i64) : (i64, i64) -> i64
// CHECK-GENERIC: "wasmssa.add"(%f32, %f32) : (f32, f32) -> f32
// CHECK-GENERIC: "wasmssa.add"(%f64, %f64) : (f64, f64) -> f64
// CHECK-GENERIC: "wasmssa.and"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.div"(%f32, %f32) : (f32, f32) -> f32
// CHECK-GENERIC: "wasmssa.div_ui"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.div_si"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.mul"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.or"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.sub"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.rem_ui"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.rem_si"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.xor"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.min"(%f32, %f32) : (f32, f32) -> f32
// CHECK-GENERIC: "wasmssa.max"(%f32, %f32) : (f32, f32) -> f32
// CHECK-GENERIC: "wasmssa.copysign"(%f32, %f32) : (f32, f32) -> f32
