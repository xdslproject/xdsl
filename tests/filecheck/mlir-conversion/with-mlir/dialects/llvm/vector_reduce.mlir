// RUN: MLIR_GENERIC_ROUNDTRIP

%arg0, %arg1 = "test.op"() : () -> (f32, vector<4xf32>)
// CHECK: [[arg0:%\d+]], [[arg1:%\d+]]

%scalar_f64 = "test.op"() : () -> f64
// CHECK: [[scalar_f64:%\d+]] = "test.op"
%vec_f64 = "test.op"() : () -> vector<2xf64>
// CHECK: [[vec_f64:%\d+]] = "test.op"

%reduce_fadd_f32 = llvm.intr.vector.reduce.fadd(%arg0, %arg1) : vector<4xf32>
// CHECK: llvm.intr.vector.reduce.fadd([[arg0]], [[arg1]]) : vector<4xf32>

%reduce_fadd_f64 = llvm.intr.vector.reduce.fadd(%scalar_f64, %vec_f64) : vector<2xf64>
// CHECK: llvm.intr.vector.reduce.fadd([[scalar_f64]], [[vec_f64]]) : vector<2xf64>

%reduce_fmul_f32 = llvm.intr.vector.reduce.fmul(%arg0, %arg1) : vector<4xf32>
// CHECK: llvm.intr.vector.reduce.fmul([[arg0]], [[arg1]]) : vector<4xf32>

%reduce_fmul_f64 = llvm.intr.vector.reduce.fmul(%scalar_f64, %vec_f64) : vector<2xf64>
// CHECK: llvm.intr.vector.reduce.fmul([[scalar_f64]], [[vec_f64]]) : vector<2xf64>
