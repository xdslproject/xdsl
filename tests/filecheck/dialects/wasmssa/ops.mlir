// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%i32 = "test.op"() : () -> i32
%i64 = "test.op"() : () -> i64
%f32 = "test.op"() : () -> f32
%f64 = "test.op"() : () -> f64

// CHECK: %i32_sum = wasmssa.add %i32 %i32 : i32
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
