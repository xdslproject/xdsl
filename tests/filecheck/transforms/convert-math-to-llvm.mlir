// RUN: xdsl-opt -p convert-math-to-llvm %s | filecheck %s
// RUN: xdsl-opt -p convert-math-to-llvm{approximate_log1p=false} %s | filecheck %s --check-prefix=NO-LOG1P

// CHECK-LABEL: func.func @intrinsics
func.func @intrinsics(%a : f32, %b : f32, %c : f32, %d : f64) {
  // CHECK-NEXT: %absf = llvm.intr.fabs(%a) : (f32) -> f32
  %absf = math.absf %a : f32
  // CHECK-NEXT: %ceil = llvm.intr.ceil(%a) : (f32) -> f32
  %ceil = math.ceil %a : f32
  // CHECK-NEXT: %copysign = llvm.intr.copysign(%a, %b) : (f32, f32) -> f32
  %copysign = math.copysign %a, %b : f32
  // CHECK-NEXT: %cos = llvm.intr.cos(%a) : (f32) -> f32
  %cos = math.cos %a : f32
  // CHECK-NEXT: %exp2 = llvm.intr.exp2(%a) : (f32) -> f32
  %exp2 = math.exp2 %a : f32
  // CHECK-NEXT: %exp = llvm.intr.exp(%a) : (f32) -> f32
  %exp = math.exp %a : f32
  // CHECK-NEXT: %floor = llvm.intr.floor(%a) : (f32) -> f32
  %floor = math.floor %a : f32
  // CHECK-NEXT: %fma = llvm.intr.fma(%a, %b, %c) : (f32, f32, f32) -> f32
  %fma = math.fma %a, %b, %c : f32
  // CHECK-NEXT: %log2 = llvm.intr.log2(%a) : (f32) -> f32
  %log2 = math.log2 %a : f32
  // CHECK-NEXT: %log = llvm.intr.log(%a) : (f32) -> f32
  %log = math.log %a : f32
  // CHECK-NEXT: %powf = llvm.intr.pow(%a, %b) : (f32, f32) -> f32
  %powf = math.powf %a, %b : f32
  // CHECK-NEXT: %sin = llvm.intr.sin(%a) : (f32) -> f32
  %sin = math.sin %a : f32
  // CHECK-NEXT: %sqrt = llvm.intr.sqrt(%a) : (f32) -> f32
  %sqrt = math.sqrt %a : f32
  // CHECK-NEXT: %sqrt_f64 = llvm.intr.sqrt(%d) : (f64) -> f64
  %sqrt_f64 = math.sqrt %d : f64
  // CHECK-NEXT: "test.op"(%absf, %ceil, %copysign, %cos, %exp2, %exp, %floor, %fma, %log2, %log, %powf, %sin, %sqrt, %sqrt_f64) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f64) -> ()
  "test.op"(%absf, %ceil, %copysign, %cos, %exp2, %exp, %floor, %fma, %log2, %log, %powf, %sin, %sqrt, %sqrt_f64) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f64) -> ()
  func.return
}

// CHECK-LABEL: func.func @intrinsics_fastmath
func.func @intrinsics_fastmath(%a : f32, %b : f32, %c : f32) {
  // CHECK-NEXT: %absf = llvm.intr.fabs(%a) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %absf = math.absf %a fastmath<fast> : f32
  // CHECK-NEXT: %powf = llvm.intr.pow(%a, %b) {fastmathFlags = #llvm.fastmath<nnan,ninf>} : (f32, f32) -> f32
  %powf = math.powf %a, %b fastmath<nnan,ninf> : f32
  // CHECK-NEXT: %fma = llvm.intr.fma(%a, %b, %c) {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32, f32) -> f32
  %fma = math.fma %a, %b, %c fastmath<contract> : f32
  // CHECK-NEXT: "test.op"(%absf, %powf, %fma) : (f32, f32, f32) -> ()
  "test.op"(%absf, %powf, %fma) : (f32, f32, f32) -> ()
  func.return
}

// CHECK-LABEL: func.func @intrinsics_vector
func.func @intrinsics_vector(%a : vector<4xf32>, %b : vector<[4]xf64>) {
  // CHECK-NEXT: %exp = llvm.intr.exp(%a) : (vector<4xf32>) -> vector<4xf32>
  %exp = math.exp %a : vector<4xf32>
  // CHECK-NEXT: %cos = llvm.intr.cos(%b) : (vector<[4]xf64>) -> vector<[4]xf64>
  %cos = math.cos %b : vector<[4]xf64>
  // CHECK-NEXT: "test.op"(%exp, %cos) : (vector<4xf32>, vector<[4]xf64>) -> ()
  "test.op"(%exp, %cos) : (vector<4xf32>, vector<[4]xf64>) -> ()
  func.return
}

// CHECK-LABEL: func.func @expm1
func.func @expm1(%a : f32, %b : vector<4xf32>) {
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK-NEXT: [[EXP:%.+]] = llvm.intr.exp(%a) : (f32) -> f32
  // CHECK-NEXT: [[EXPM1:%.+]] = llvm.fsub [[EXP]], [[ONE]] : f32
  %expm1 = math.expm1 %a : f32
  // CHECK-NEXT: [[ONE_VEC:%.+]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK-NEXT: [[EXP_VEC:%.+]] = llvm.intr.exp(%b) {fastmathFlags = #llvm.fastmath<fast>} : (vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: [[EXPM1_VEC:%.+]] = llvm.fsub [[EXP_VEC]], [[ONE_VEC]] {fastmathFlags = #llvm.fastmath<fast>} : vector<4xf32>
  %expm1_vec = math.expm1 %b fastmath<fast> : vector<4xf32>
  // CHECK-NEXT: "test.op"([[EXPM1]], [[EXPM1_VEC]]) : (f32, vector<4xf32>) -> ()
  "test.op"(%expm1, %expm1_vec) : (f32, vector<4xf32>) -> ()
  func.return
}

// CHECK-LABEL: func.func @log1p
// NO-LOG1P-LABEL: func.func @log1p
func.func @log1p(%a : f64, %b : vector<[2]xf64>) {
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
  // CHECK-NEXT: [[ADD:%.+]] = llvm.fadd [[ONE]], %a : f64
  // CHECK-NEXT: [[LOG1P:%.+]] = llvm.intr.log([[ADD]]) : (f64) -> f64
  // NO-LOG1P-NEXT: %log1p = math.log1p %a : f64
  %log1p = math.log1p %a : f64
  // CHECK-NEXT: [[ONE_VEC:%.+]] = llvm.mlir.constant(dense<1.000000e+00> : vector<[2]xf64>) : vector<[2]xf64>
  // CHECK-NEXT: [[ADD_VEC:%.+]] = llvm.fadd [[ONE_VEC]], %b {fastmathFlags = #llvm.fastmath<afn>} : vector<[2]xf64>
  // CHECK-NEXT: [[LOG1P_VEC:%.+]] = llvm.intr.log([[ADD_VEC]]) {fastmathFlags = #llvm.fastmath<afn>} : (vector<[2]xf64>) -> vector<[2]xf64>
  // NO-LOG1P-NEXT: %log1p_vec = math.log1p %b fastmath<afn> : vector<[2]xf64>
  %log1p_vec = math.log1p %b fastmath<afn> : vector<[2]xf64>
  // CHECK-NEXT: "test.op"([[LOG1P]], [[LOG1P_VEC]]) : (f64, vector<[2]xf64>) -> ()
  "test.op"(%log1p, %log1p_vec) : (f64, vector<[2]xf64>) -> ()
  func.return
}

// CHECK-LABEL: func.func @rsqrt
func.func @rsqrt(%a : f16, %b : vector<8xbf16>) {
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1.000000e+00 : f16) : f16
  // CHECK-NEXT: [[SQRT:%.+]] = llvm.intr.sqrt(%a) {fastmathFlags = #llvm.fastmath<arcp>} : (f16) -> f16
  // CHECK-NEXT: [[RSQRT:%.+]] = llvm.fdiv [[ONE]], [[SQRT]] {fastmathFlags = #llvm.fastmath<arcp>} : f16
  %rsqrt = math.rsqrt %a fastmath<arcp> : f16
  // CHECK-NEXT: [[ONE_VEC:%.+]] = llvm.mlir.constant(dense<1.000000e+00> : vector<8xbf16>) : vector<8xbf16>
  // CHECK-NEXT: [[SQRT_VEC:%.+]] = llvm.intr.sqrt(%b) : (vector<8xbf16>) -> vector<8xbf16>
  // CHECK-NEXT: [[RSQRT_VEC:%.+]] = llvm.fdiv [[ONE_VEC]], [[SQRT_VEC]] : vector<8xbf16>
  %rsqrt_vec = math.rsqrt %b : vector<8xbf16>
  // CHECK-NEXT: "test.op"([[RSQRT]], [[RSQRT_VEC]]) : (f16, vector<8xbf16>) -> ()
  "test.op"(%rsqrt, %rsqrt_vec) : (f16, vector<8xbf16>) -> ()
  func.return
}

// Constants of f80 and f128 cannot be encoded yet, so only operations that do not
// need one are converted.

// CHECK-LABEL: func.func @extended_precision
func.func @extended_precision(%a : f80, %b : f128, %c : vector<2xf80>) {
  // CHECK-NEXT: %exp = llvm.intr.exp(%a) : (f80) -> f80
  %exp = math.exp %a : f80
  // CHECK-NEXT: %sqrt = llvm.intr.sqrt(%b) : (f128) -> f128
  %sqrt = math.sqrt %b : f128
  // CHECK-NEXT: %exp_vec = llvm.intr.exp(%c) : (vector<2xf80>) -> vector<2xf80>
  %exp_vec = math.exp %c : vector<2xf80>
  // CHECK-NEXT: %rsqrt = math.rsqrt %a : f80
  %rsqrt = math.rsqrt %a : f80
  // CHECK-NEXT: %expm1 = math.expm1 %b : f128
  %expm1 = math.expm1 %b : f128
  // CHECK-NEXT: %log1p_vec = math.log1p %c : vector<2xf80>
  %log1p_vec = math.log1p %c : vector<2xf80>
  // CHECK-NEXT: "test.op"(%exp, %sqrt, %exp_vec, %rsqrt, %expm1, %log1p_vec) : (f80, f128, vector<2xf80>, f80, f128, vector<2xf80>) -> ()
  "test.op"(%exp, %sqrt, %exp_vec, %rsqrt, %expm1, %log1p_vec) : (f80, f128, vector<2xf80>, f80, f128, vector<2xf80>) -> ()
  func.return
}

// Discardable attributes are forwarded to the converted operations, but not to the
// materialized constants.

// CHECK-LABEL: func.func @discardable_attributes
func.func @discardable_attributes(%a : f32) {
  // CHECK-NEXT: %sqrt = llvm.intr.sqrt(%a) {test.attr} : (f32) -> f32
  %sqrt = math.sqrt %a {test.attr} : f32
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK-NEXT: [[SQRT:%.+]] = llvm.intr.sqrt(%a) {test.attr} : (f32) -> f32
  // CHECK-NEXT: [[RSQRT:%.+]] = llvm.fdiv [[ONE]], [[SQRT]] {test.attr} : f32
  %rsqrt = math.rsqrt %a {test.attr} : f32
  // CHECK-NEXT: "test.op"(%sqrt, [[RSQRT]]) : (f32, f32) -> ()
  "test.op"(%sqrt, %rsqrt) : (f32, f32) -> ()
  func.return
}

// Operations without an LLVM lowering, or on types that are not LLVM-compatible floats
// or one-dimensional vectors of them, are left unchanged.

// CHECK-LABEL: func.func @unsupported
func.func @unsupported(%a : f32, %b : vector<2x4xf32>, %c : vector<f32>, %d : tensor<4xf32>, %e : f8E4M3FN) {
  // CHECK-NEXT: %tan = math.tan %a : f32
  %tan = math.tan %a : f32
  // CHECK-NEXT: %exp_2d = math.exp %b : vector<2x4xf32>
  %exp_2d = math.exp %b : vector<2x4xf32>
  // CHECK-NEXT: %exp_0d = math.exp %c : vector<f32>
  %exp_0d = math.exp %c : vector<f32>
  // CHECK-NEXT: %exp_tensor = math.exp %d : tensor<4xf32>
  %exp_tensor = math.exp %d : tensor<4xf32>
  // CHECK-NEXT: %exp_f8 = math.exp %e : f8E4M3FN
  %exp_f8 = math.exp %e : f8E4M3FN
  // CHECK-NEXT: %rsqrt_2d = math.rsqrt %b : vector<2x4xf32>
  %rsqrt_2d = math.rsqrt %b : vector<2x4xf32>
  // CHECK-NEXT: "test.op"(%tan, %exp_2d, %exp_0d, %exp_tensor, %exp_f8, %rsqrt_2d) : (f32, vector<2x4xf32>, vector<f32>, tensor<4xf32>, f8E4M3FN, vector<2x4xf32>) -> ()
  "test.op"(%tan, %exp_2d, %exp_0d, %exp_tensor, %exp_f8, %rsqrt_2d) : (f32, vector<2x4xf32>, vector<f32>, tensor<4xf32>, f8E4M3FN, vector<2x4xf32>) -> ()
  func.return
}
