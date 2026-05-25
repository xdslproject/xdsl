// REQUIRES: llvm-diff
// RUN: xdsl-opt -t llvm %s > %t.xdsl.ll
// RUN: mlir-translate --mlir-to-llvmir %s > %t.mlir.ll
// RUN: %llvm-diff %t.xdsl.ll %t.mlir.ll

module {
  llvm.func @reduce_fadd_f32(%arg0: f32, %arg1: vector<4xf32>) -> f32 {
    %0 = "llvm.intr.vector.reduce.fadd"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
    llvm.return %0 : f32
  }

  llvm.func @reduce_fadd_f64(%arg0: f64, %arg1: vector<2xf64>) -> f64 {
    %0 = "llvm.intr.vector.reduce.fadd"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<2xf64>) -> f64
    llvm.return %0 : f64
  }

  llvm.func @reduce_fmul_f32(%arg0: f32, %arg1: vector<4xf32>) -> f32 {
    %0 = "llvm.intr.vector.reduce.fmul"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
    llvm.return %0 : f32
  }

  llvm.func @reduce_fmul_f64(%arg0: f64, %arg1: vector<2xf64>) -> f64 {
    %0 = "llvm.intr.vector.reduce.fmul"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<2xf64>) -> f64
    llvm.return %0 : f64
  }
}
