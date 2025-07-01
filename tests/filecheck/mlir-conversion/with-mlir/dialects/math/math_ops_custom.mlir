// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%vali32 = "test.op"() : () -> i32
%vali64 = "test.op"() : () -> i64
%valf32 = "test.op"() : () -> f32
%valf64 = "test.op"() : () -> f64
%vec_vali64 = "test.op"() : () -> vector<4xi64>
%vec_valf64 = "test.op"() : () -> vector<4xf64>

// CHECK:      [[VALI32:%.*]] = "test.op"() : () -> i32
// CHECK-NEXT: [[VALI64:%.*]] = "test.op"() : () -> i64
// CHECK-NEXT: [[VALF32:%.*]] = "test.op"() : () -> f32
// CHECK-NEXT: [[VALF64:%.*]] = "test.op"() : () -> f64
// CHECK-NEXT: [[VEC_VALI64:%.*]] = "test.op"() : () -> vector<4xi64>
// CHECK-NEXT: [[VEC_VALF64:%.*]] = "test.op"() : () -> vector<4xf64>

%rhsi32 = "test.op"() : () -> i32
%rhsi64 = "test.op"() : () -> i64
%rhsf32 = "test.op"() : () -> f32
%rhsf64 = "test.op"() : () -> f64
%vec_rhsi64 = "test.op"() : () -> vector<4xi64>
%vec_rhsf64 = "test.op"() : () -> vector<4xf64>

// CHECK-NEXT: [[RHSI32:%.*]] = "test.op"() : () -> i32
// CHECK-NEXT: [[RHSI64:%.*]] = "test.op"() : () -> i64
// CHECK-NEXT: [[RHSF32:%.*]] = "test.op"() : () -> f32
// CHECK-NEXT: [[RHSF64:%.*]] = "test.op"() : () -> f64
// CHECK-NEXT: [[VEC_RHSI64:%.*]] = "test.op"() : () -> vector<4xi64>
// CHECK-NEXT: [[VEC_RHSF64:%.*]] = "test.op"() : () -> vector<4xf64>

%absf0 = math.absf %valf32 : f32
%absf1 = math.absf %valf64 : f64
%vabsf1 = math.absf %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.absf [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.absf [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.absf [[VEC_VALF64]] : vector<4xf64>

%absi0 = math.absi %vali32: i32
%absi1 = math.absi %vali64: i64
%vabsi1 = math.absi %vec_vali64 : vector<4xi64>

// CHECK-NEXT: {{%.*}} = math.absi [[VALI32]] : i32
// CHECK-NEXT: {{%.*}} = math.absi [[VALI64]] : i64
// CHECK-NEXT: {{%.*}} = math.absi [[VEC_VALI64]] : vector<4xi64>

%atan2f0 = math.atan2 %valf32, %rhsf32 : f32
%atan2f1 = math.atan2 %valf64, %rhsf64 : f64
%vatan2f1 = math.atan2 %vec_valf64, %vec_valf64: vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.atan2 [[VALF32]], [[RHSF32]] : f32
// CHECK-NEXT: {{%.*}} = math.atan2 [[VALF64]], [[RHSF64]] : f64
// CHECK-NEXT: {{%.*}} = math.atan2 [[VEC_VALF64]], [[VEC_VALF64]] : vector<4xf64>

%atanf0 = math.atan %valf32 : f32
%atanf1 = math.atan %valf64 : f64
%vatanf1 = math.atan %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.atan [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.atan [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.atan [[VEC_VALF64]] : vector<4xf64>

%cbrtf0 = math.cbrt %valf32 : f32
%cbrtf1 = math.cbrt %valf64 : f64
%vcbrtf1 = math.cbrt %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.cbrt [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.cbrt [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.cbrt [[VEC_VALF64]] : vector<4xf64>

%ceilf0 = math.ceil %valf32 : f32
%ceilf1 = math.ceil %valf64 : f64
%vceilf1 = math.ceil %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.ceil [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.ceil [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.ceil [[VEC_VALF64]] : vector<4xf64>

%copysign0 = math.copysign %valf32, %rhsf32 : f32
%copysign1 = math.copysign %valf64, %rhsf64 : f64
%vcopysign1 = math.copysign %vec_valf64, %vec_rhsf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.copysign [[VALF32]], [[RHSF32]] : f32
// CHECK-NEXT: {{%.*}} = math.copysign [[VALF64]], [[RHSF64]] : f64
// CHECK-NEXT: {{%.*}} = math.copysign [[VEC_VALF64]], [[VEC_RHSF64]] : vector<4xf64>

%cosf0 = math.cos %valf32 : f32
%cosf1 = math.cos %valf64 : f64
%vcosf1 = math.cos %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.cos [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.cos [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.cos [[VEC_VALF64]] : vector<4xf64>

%coshf0 = math.cosh %valf32 : f32
%coshf1 = math.cosh %valf64 : f64
%vcoshf1 = math.cosh %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.cosh [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.cosh [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.cosh [[VEC_VALF64]] : vector<4xf64>

%ctlzi0 = math.ctlz %vali32 : i32
%ctlzi1 = math.ctlz %vali64 : i64
%vctlzi1 = math.ctlz %vec_vali64 : vector<4xi64>

// CHECK-NEXT: {{%.*}} = math.ctlz [[VALI32]] : i32
// CHECK-NEXT: {{%.*}} = math.ctlz [[VALI64]] : i64
// CHECK-NEXT: {{%.*}} = math.ctlz [[VEC_VALI64]] : vector<4xi64>

%cttzi0 = math.cttz %vali32 : i32
%cttzi1 = math.cttz %vali64 : i64
%vcttzi1 = math.cttz %vec_vali64 : vector<4xi64>

// CHECK-NEXT: {{%.*}} = math.cttz [[VALI32]] : i32
// CHECK-NEXT: {{%.*}} = math.cttz [[VALI64]] : i64
// CHECK-NEXT: {{%.*}} = math.cttz [[VEC_VALI64]] : vector<4xi64>

%ctpopi0 = math.ctpop %vali32 : i32
%ctpopi1 = math.ctpop %vali64 : i64
%vctpopi1 = math.ctpop %vec_vali64 : vector<4xi64>

// CHECK-NEXT: {{%.*}} = math.ctpop [[VALI32]] : i32
// CHECK-NEXT: {{%.*}} = math.ctpop [[VALI64]] : i64
// CHECK-NEXT: {{%.*}} = math.ctpop [[VEC_VALI64]] : vector<4xi64>

%erff0 = math.erf %valf32 : f32
%erff1 = math.erf %valf64 : f64
%verff1 = math.erf %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.erf [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.erf [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.erf [[VEC_VALF64]] : vector<4xf64>

%exp2f0 = math.exp2 %valf32 : f32
%exp2f1 = math.exp2 %valf64 : f64
%vexp2f1 = math.exp2 %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.exp2 [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.exp2 [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.exp2 [[VEC_VALF64]] : vector<4xf64>

%expm10 = math.expm1 %valf32 : f32
%expm11 = math.expm1 %valf64 : f64
%vexpm11 = math.expm1 %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.expm1 [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.expm1 [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.expm1 [[VEC_VALF64]] : vector<4xf64>

%exp0 = math.exp %valf32 : f32
%exp1 = math.exp %valf64 : f64
%vexp1 = math.exp %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.exp [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.exp [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.exp [[VEC_VALF64]] : vector<4xf64>

%fpowi0 = math.fpowi %valf32, %vali32 : f32, i32
%fpowi1 = math.fpowi %valf32, %vali64 : f32, i64
%fpowi2 = math.fpowi %valf64, %vali32 : f64, i32
%fpowi3 = math.fpowi %valf64, %vali64 : f64, i64
%vfpowi3 = math.fpowi %vec_valf64, %vec_vali64 : vector<4xf64>, vector<4xi64>

// CHECK-NEXT: {{%.*}} = math.fpowi [[VALF32]], [[VALI32]] : f32, i32
// CHECK-NEXT: {{%.*}} = math.fpowi [[VALF32]], [[VALI64]] : f32, i64
// CHECK-NEXT: {{%.*}} = math.fpowi [[VALF64]], [[VALI32]] : f64, i32
// CHECK-NEXT: {{%.*}} = math.fpowi [[VALF64]], [[VALI64]] : f64, i64
// CHECK-NEXT: {{%.*}} = math.fpowi [[VEC_VALF64]], [[VEC_VALI64]] : vector<4xf64>, vector<4xi64>

%floor0 = math.floor %valf32 : f32
%floor1 = math.floor %valf64 : f64
%vfloor1 = math.floor %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.floor [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.floor [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.floor [[VEC_VALF64]] : vector<4xf64>

%fma0 = math.fma %valf32, %valf32, %rhsf32 : f32
%fma1 = math.fma %valf64, %valf64, %rhsf64 : f64
%vfma1 = math.fma %vec_valf64, %vec_valf64, %vec_rhsf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.fma [[VALF32]], [[VALF32]], [[RHSF32]] : f32
// CHECK-NEXT: {{%.*}} = math.fma [[VALF64]], [[VALF64]], [[RHSF64]] : f64
// CHECK-NEXT: {{%.*}} = math.fma [[VEC_VALF64]], [[VEC_VALF64]], [[VEC_RHSF64]] : vector<4xf64>

%ipowi0 = math.ipowi %vali32, %rhsi32 : i32
%ipowi1 = math.ipowi %vali64, %rhsi64 : i64
%vipowi1 = math.ipowi %vec_vali64, %vec_rhsi64 : vector<4xi64>

// CHECK-NEXT: {{%.*}} = math.ipowi [[VALI32]], [[RHSI32]] : i32
// CHECK-NEXT: {{%.*}} = math.ipowi [[VALI64]], [[RHSI64]] : i64
// CHECK-NEXT: {{%.*}} = math.ipowi [[VEC_VALI64]], [[VEC_RHSI64]] : vector<4xi64>

%log100 = math.log10 %valf32 : f32
%log101 = math.log10 %valf64 : f64
%vlog101 = math.log10 %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.log10 [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.log10 [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.log10 [[VEC_VALF64]] : vector<4xf64>

%log1p0 = math.log1p %valf32 : f32
%log1p1 = math.log1p %valf64 : f64
%vlog1p1 = math.log1p %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.log1p [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.log1p [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.log1p [[VEC_VALF64]] : vector<4xf64>

%log20 = math.log2 %valf32 : f32
%log21 = math.log2 %valf64 : f64
%vlog21 = math.log2 %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.log2 [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.log2 [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.log2 [[VEC_VALF64]] : vector<4xf64>

%log0 = math.log %valf32 : f32
%log1 = math.log %valf64 : f64
%vlog1 = math.log %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.log [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.log [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.log [[VEC_VALF64]] : vector<4xf64>

%powf0 = math.powf %valf32, %rhsf32 : f32
%powf1 = math.powf %valf64, %rhsf64 : f64
%vpowf1 = math.powf %vec_valf64, %vec_rhsf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.powf [[VALF32]], [[RHSF32]] : f32
// CHECK-NEXT: {{%.*}} = math.powf [[VALF64]], [[RHSF64]] : f64
// CHECK-NEXT: {{%.*}} = math.powf [[VEC_VALF64]], [[VEC_RHSF64]] : vector<4xf64>

%roundeven0 = math.roundeven %valf32 : f32
%roundeven1 = math.roundeven %valf64 : f64
%vroundeven1 = math.roundeven %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.roundeven [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.roundeven [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.roundeven [[VEC_VALF64]] : vector<4xf64>

%round0 = math.round %valf32 : f32
%round1 = math.round %valf64 : f64
%vround1 = math.round %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.round [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.round [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.round [[VEC_VALF64]] : vector<4xf64>

%rsqrt0 = math.rsqrt %valf32 : f32
%rsqrt1 = math.rsqrt %valf64 : f64
%vrsqrt1 = math.rsqrt %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.rsqrt [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.rsqrt [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.rsqrt [[VEC_VALF64]] : vector<4xf64>

%sin0 = math.sin %valf32 : f32
%sin1 = math.sin %valf64 : f64
%vsin1 = math.sin %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.sin [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.sin [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.sin [[VEC_VALF64]] : vector<4xf64>

%sinh0 = math.sinh %valf32 : f32
%sinh1 = math.sinh %valf64 : f64
%vsinh1 = math.sinh %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.sinh [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.sinh [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.sinh [[VEC_VALF64]] : vector<4xf64>

%sqrt0 = math.sqrt %valf32 : f32
%sqrt1 = math.sqrt %valf64 : f64
%vsqrt1 = math.sqrt %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.sqrt [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.sqrt [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.sqrt [[VEC_VALF64]] : vector<4xf64>

%tan0 = math.tan %valf32 : f32
%tan1 = math.tan %valf64 : f64
%vtan1 = math.tan %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.tan [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.tan [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.tan [[VEC_VALF64]] : vector<4xf64>

%tanh0 = math.tanh %valf32 : f32
%tanh1 = math.tanh %valf64 : f64
%vtanh1 = math.tanh %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.tanh [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.tanh [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.tanh [[VEC_VALF64]] : vector<4xf64>

%trunc0 = math.trunc %valf32 : f32
%trunc1 = math.trunc %valf64 : f64
%vtrunc1 = math.trunc %vec_valf64 : vector<4xf64>

// CHECK-NEXT: {{%.*}} = math.trunc [[VALF32]] : f32
// CHECK-NEXT: {{%.*}} = math.trunc [[VALF64]] : f64
// CHECK-NEXT: {{%.*}} = math.trunc [[VEC_VALF64]] : vector<4xf64>
