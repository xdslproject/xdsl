// RUN: XDSL_ROUNDTRIP

%arg0 = "test.op"() : () -> f32
%arg1 = "test.op"() : () -> f64
%arg2 = "test.op"() : () -> vector<4xf32>

%0 = llvm.intr.fabs(%arg0) : (f32) -> f32
// CHECK: %0 = llvm.intr.fabs(%arg0) : (f32) -> f32

%1 = llvm.intr.fabs(%arg1) : (f64) -> f64
// CHECK-NEXT: %1 = llvm.intr.fabs(%arg1) : (f64) -> f64

%2 = llvm.intr.fabs(%arg2) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %2 = llvm.intr.fabs(%arg2) : (vector<4xf32>) -> vector<4xf32>
