// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

builtin.module {
    %arg0, %arg1, %arg2 = "test.op"() : () -> (f32, f64, vector<4xf32>)
    // CHECK: [[arg0:%\d+]], [[arg1:%\d+]], [[arg2:%\d+]]

    %0 = llvm.intr.fabs(%arg0) : (f32) -> f32
    // CHECK: llvm.intr.fabs([[arg0]]) : (f32) -> f32

    %1 = llvm.intr.fabs(%arg1) : (f64) -> f64
    // CHECK: llvm.intr.fabs([[arg1]]) : (f64) -> f64

    %2 = llvm.intr.fabs(%arg2) : (vector<4xf32>) -> vector<4xf32>
    // CHECK: llvm.intr.fabs([[arg2]]) : (vector<4xf32>) -> vector<4xf32>
}
