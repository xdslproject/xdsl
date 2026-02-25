// RUN: xdsl-opt --print-debuginfo %s | FileCheck %s

// Test function with argument locations
func.func @test(%arg0: i32 loc("arg.mlir":1:1), %arg1: i64 loc("arg.mlir":2:2)) loc("func.mlir":10:8) {
    // CHECK: func.func @test(%arg0 : i32 loc("arg.mlir":1:1), %arg1 : i64 loc("arg.mlir":2:2)) loc("func.mlir":10:8)
    func.return
}
