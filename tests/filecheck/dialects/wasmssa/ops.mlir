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

// CHECK-NEXT: %i32_eq = wasmssa.eq %i32 %i32 : i32 -> i32
%i32_eq = wasmssa.eq %i32 %i32 : i32 -> i32
// CHECK-NEXT: %f32_ne = wasmssa.ne %f32 %f32 : f32 -> i32
%f32_ne = wasmssa.ne %f32 %f32 : f32 -> i32
// CHECK-NEXT: %i32_lt_si = wasmssa.lt_si %i32 %i32 : i32 -> i32
%i32_lt_si = wasmssa.lt_si %i32 %i32 : i32 -> i32
// CHECK-NEXT: %i32_lt_ui = wasmssa.lt_ui %i32 %i32 : i32 -> i32
%i32_lt_ui = wasmssa.lt_ui %i32 %i32 : i32 -> i32
// CHECK-NEXT: %i32_le_si = wasmssa.le_si %i32 %i32 : i32 -> i32
%i32_le_si = wasmssa.le_si %i32 %i32 : i32 -> i32
// CHECK-NEXT: %i32_le_ui = wasmssa.le_ui %i32 %i32 : i32 -> i32
%i32_le_ui = wasmssa.le_ui %i32 %i32 : i32 -> i32
// CHECK-NEXT: %i64_gt_si = wasmssa.gt_si %i64 %i64 : i64 -> i32
%i64_gt_si = wasmssa.gt_si %i64 %i64 : i64 -> i32
// CHECK-NEXT: %i64_gt_ui = wasmssa.gt_ui %i64 %i64 : i64 -> i32
%i64_gt_ui = wasmssa.gt_ui %i64 %i64 : i64 -> i32
// CHECK-NEXT: %i64_ge_si = wasmssa.ge_si %i64 %i64 : i64 -> i32
%i64_ge_si = wasmssa.ge_si %i64 %i64 : i64 -> i32
// CHECK-NEXT: %i64_ge_ui = wasmssa.ge_ui %i64 %i64 : i64 -> i32
%i64_ge_ui = wasmssa.ge_ui %i64 %i64 : i64 -> i32
// CHECK-NEXT: %f32_lt = wasmssa.lt %f32 %f32 : f32 -> i32
%f32_lt = wasmssa.lt %f32 %f32 : f32 -> i32
// CHECK-NEXT: %f32_le = wasmssa.le %f32 %f32 : f32 -> i32
%f32_le = wasmssa.le %f32 %f32 : f32 -> i32
// CHECK-NEXT: %f64_gt = wasmssa.gt %f64 %f64 : f64 -> i32
%f64_gt = wasmssa.gt %f64 %f64 : f64 -> i32
// CHECK-NEXT: %f64_ge = wasmssa.ge %f64 %f64 : f64 -> i32
%f64_ge = wasmssa.ge %f64 %f64 : f64 -> i32
// CHECK-NEXT: %i64_eqz = wasmssa.eqz %i64 : i64 -> i32
%i64_eqz = wasmssa.eqz %i64 : i64 -> i32

// CHECK-NEXT: %f32_abs = wasmssa.abs %f32 : f32
%f32_abs = wasmssa.abs %f32 : f32
// CHECK-NEXT: %f32_ceil = wasmssa.ceil %f32 : f32
%f32_ceil = wasmssa.ceil %f32 : f32
// CHECK-NEXT: %f32_floor = wasmssa.floor %f32 : f32
%f32_floor = wasmssa.floor %f32 : f32
// CHECK-NEXT: %f32_neg = wasmssa.neg %f32 : f32
%f32_neg = wasmssa.neg %f32 : f32
// CHECK-NEXT: %f32_sqrt = wasmssa.sqrt %f32 : f32
%f32_sqrt = wasmssa.sqrt %f32 : f32
// CHECK-NEXT: %f32_trunc = wasmssa.trunc %f32 : f32
%f32_trunc = wasmssa.trunc %f32 : f32
// CHECK-NEXT: %i32_clz = wasmssa.clz %i32 : i32
%i32_clz = wasmssa.clz %i32 : i32
// CHECK-NEXT: %i32_ctz = wasmssa.ctz %i32 : i32
%i32_ctz = wasmssa.ctz %i32 : i32
// CHECK-NEXT: %i32_popcnt = wasmssa.popcnt %i32 : i32
%i32_popcnt = wasmssa.popcnt %i32 : i32
// CHECK-NEXT: %f32_convert_s = wasmssa.convert_s %i32 : i32 to f32
%f32_convert_s = wasmssa.convert_s %i32 : i32 to f32
// CHECK-NEXT: %f64_convert_u = wasmssa.convert_u %i64 : i64 to f64
%f64_convert_u = wasmssa.convert_u %i64 : i64 to f64
// CHECK-NEXT: %f32_demote = wasmssa.demote %f64 : f64 to f32
%f32_demote = wasmssa.demote %f64 : f64 to f32
// CHECK-NEXT: %i64_extend_i32_s = wasmssa.extend_i32_s %i32 to i64
%i64_extend_i32_s = wasmssa.extend_i32_s %i32 to i64
// CHECK-NEXT: %i64_extend_i32_u = wasmssa.extend_i32_u %i32 to i64
%i64_extend_i32_u = wasmssa.extend_i32_u %i32 to i64
// CHECK-NEXT: %i32_extend_eight = wasmssa.extend 8 : i64 low bits from %i32 : i32
%i32_extend_eight = wasmssa.extend 8 : i64 low bits from %i32 : i32
// CHECK-NEXT: %i32_extend_sixteen = wasmssa.extend 16 : i64 low bits from %i32 : i32
%i32_extend_sixteen = wasmssa.extend 16 : i64 low bits from %i32 : i32
// CHECK-NEXT: %i64_extend_eight = wasmssa.extend 8 : i64 low bits from %i64 : i64
%i64_extend_eight = wasmssa.extend 8 : i64 low bits from %i64 : i64
// CHECK-NEXT: %i64_extend_sixteen = wasmssa.extend 16 : i64 low bits from %i64 : i64
%i64_extend_sixteen = wasmssa.extend 16 : i64 low bits from %i64 : i64
// CHECK-NEXT: %i64_extend_thirty_two = wasmssa.extend 32 : i64 low bits from %i64 : i64
%i64_extend_thirty_two = wasmssa.extend 32 : i64 low bits from %i64 : i64
// CHECK-NEXT: %f64_promote = wasmssa.promote %f32 : f32 to f64
%f64_promote = wasmssa.promote %f32 : f32 to f64
// CHECK-NEXT: %i32_wrap = wasmssa.wrap %i64 : i64 to i32
%i32_wrap = wasmssa.wrap %i64 : i64 to i32
// CHECK-NEXT: %f32_reinterpret = wasmssa.reinterpret %i32 : i32 as f32
%f32_reinterpret = wasmssa.reinterpret %i32 : i32 as f32

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
// CHECK-GENERIC: "wasmssa.eq"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.ne"(%f32, %f32) : (f32, f32) -> i32
// CHECK-GENERIC: "wasmssa.lt_si"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.lt_ui"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.le_si"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.le_ui"(%i32, %i32) : (i32, i32) -> i32
// CHECK-GENERIC: "wasmssa.gt_si"(%i64, %i64) : (i64, i64) -> i32
// CHECK-GENERIC: "wasmssa.gt_ui"(%i64, %i64) : (i64, i64) -> i32
// CHECK-GENERIC: "wasmssa.ge_si"(%i64, %i64) : (i64, i64) -> i32
// CHECK-GENERIC: "wasmssa.ge_ui"(%i64, %i64) : (i64, i64) -> i32
// CHECK-GENERIC: "wasmssa.lt"(%f32, %f32) : (f32, f32) -> i32
// CHECK-GENERIC: "wasmssa.le"(%f32, %f32) : (f32, f32) -> i32
// CHECK-GENERIC: "wasmssa.gt"(%f64, %f64) : (f64, f64) -> i32
// CHECK-GENERIC: "wasmssa.ge"(%f64, %f64) : (f64, f64) -> i32
// CHECK-GENERIC: "wasmssa.eqz"(%i64) : (i64) -> i32
// CHECK-GENERIC: "wasmssa.abs"(%f32) : (f32) -> f32
// CHECK-GENERIC: "wasmssa.ceil"(%f32) : (f32) -> f32
// CHECK-GENERIC: "wasmssa.floor"(%f32) : (f32) -> f32
// CHECK-GENERIC: "wasmssa.neg"(%f32) : (f32) -> f32
// CHECK-GENERIC: "wasmssa.sqrt"(%f32) : (f32) -> f32
// CHECK-GENERIC: "wasmssa.trunc"(%f32) : (f32) -> f32
// CHECK-GENERIC: "wasmssa.clz"(%i32) : (i32) -> i32
// CHECK-GENERIC: "wasmssa.ctz"(%i32) : (i32) -> i32
// CHECK-GENERIC: "wasmssa.popcnt"(%i32) : (i32) -> i32
// CHECK-GENERIC: "wasmssa.convert_s"(%i32) : (i32) -> f32
// CHECK-GENERIC: "wasmssa.convert_u"(%i64) : (i64) -> f64
// CHECK-GENERIC: "wasmssa.demote"(%f64) : (f64) -> f32
// CHECK-GENERIC: "wasmssa.extend_i32_s"(%i32) : (i32) -> i64
// CHECK-GENERIC: "wasmssa.extend_i32_u"(%i32) : (i32) -> i64
// CHECK-GENERIC: "wasmssa.extend"(%i32) <{bitsToTake = 8 : i64}> : (i32) -> i32
// CHECK-GENERIC: "wasmssa.extend"(%i32) <{bitsToTake = 16 : i64}> : (i32) -> i32
// CHECK-GENERIC: "wasmssa.extend"(%i64) <{bitsToTake = 8 : i64}> : (i64) -> i64
// CHECK-GENERIC: "wasmssa.extend"(%i64) <{bitsToTake = 16 : i64}> : (i64) -> i64
// CHECK-GENERIC: "wasmssa.extend"(%i64) <{bitsToTake = 32 : i64}> : (i64) -> i64
// CHECK-GENERIC: "wasmssa.promote"(%f32) : (f32) -> f64
// CHECK-GENERIC: "wasmssa.wrap"(%i64) : (i64) -> i32
// CHECK-GENERIC: "wasmssa.reinterpret"(%i32) : (i32) -> f32
