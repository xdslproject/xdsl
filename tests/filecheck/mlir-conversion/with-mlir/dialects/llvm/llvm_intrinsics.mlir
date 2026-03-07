// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

%arg0, %arg1, %arg2 = "test.op"() : () -> (f32, f64, vector<4xf32>)
// CHECK: [[arg0:%\d+]], [[arg1:%\d+]], [[arg2:%\d+]]

%0 = llvm.intr.fabs(%arg0) : (f32) -> f32
// CHECK: llvm.intr.fabs([[arg0]]) : (f32) -> f32

%1 = llvm.intr.fabs(%arg1) : (f64) -> f64
// CHECK: llvm.intr.fabs([[arg1]]) : (f64) -> f64

%2 = llvm.intr.fabs(%arg2) : (vector<4xf32>) -> vector<4xf32>
// CHECK: llvm.intr.fabs([[arg2]]) : (vector<4xf32>) -> vector<4xf32>

%3 = llvm.fneg %arg0 : f32
// CHECK: llvm.fneg [[arg0]] : f32

%4 = llvm.fneg %arg1 : f64
// CHECK: llvm.fneg [[arg1]] : f64

%5 = llvm.fneg %arg2 : vector<4xf32>
// CHECK: llvm.fneg [[arg2]] : vector<4xf32>

%6 = llvm.fneg %arg0 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK: llvm.fneg [[arg0]] {fastmathFlags = #llvm.fastmath<fast>} : f32

%fcmp_lhs, %fcmp_rhs = "test.op"() : () -> (f32, f32)
// CHECK: [[fcmp_lhs:%\d+]], [[fcmp_rhs:%\d+]]

%7 = llvm.fcmp "oeq" %fcmp_lhs, %fcmp_rhs : f32
// CHECK: llvm.fcmp "oeq" [[fcmp_lhs]], [[fcmp_rhs]] : f32

%8 = llvm.fcmp "ult" %fcmp_lhs, %fcmp_rhs : f32
// CHECK: llvm.fcmp "ult" [[fcmp_lhs]], [[fcmp_rhs]] : f32

%9 = llvm.fcmp "one" %fcmp_lhs, %fcmp_rhs : f32
// CHECK: llvm.fcmp "one" [[fcmp_lhs]], [[fcmp_rhs]] : f32

%ptr = "test.op"() : () -> !llvm.ptr
// CHECK: [[ptr:%\d+]] = "test.op"
%vec_val = "test.op"() : () -> vector<4xf32>
// CHECK: [[vec_val:%\d+]] = "test.op"
%vec_mask = "test.op"() : () -> vector<4xi1>
// CHECK: [[vec_mask:%\d+]] = "test.op"

llvm.intr.masked.store %vec_val, %ptr, %vec_mask {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
// CHECK: llvm.intr.masked.store [[vec_val]], [[ptr]], [[vec_mask]] {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
