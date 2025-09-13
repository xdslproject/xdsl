// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

%a = "test.op"() : () -> (vector<2xf32>)

%0 = vector.shuffle %a, %a [] : vector<2xf32>, vector<2xf32>
// CHECK: Result vector type must not be 0-D.

// -----

%a = "test.op"() : () -> (vector<2xf32>)
%0 = "vector.shuffle"(%a, %a) <{mask = array<i64: 0>}> : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
// CHECK: Operation does not verify: Length of mask array<i64: 0> must equal leading dim of result vector<2xf32>.

// -----

%a, %b = "test.op"() : () -> (vector<2xf32>, vector<f32>)
%0 = "vector.shuffle"(%a, %b) <{mask = array<i64: 0>}> : (vector<2xf32>, vector<f32>) -> vector<1xf32>
// CHECK: Operation does not verify: Inputs must either both be non-0-D or both be 0-D

// -----

%a = "test.op"() : () -> (vector<f32>)
%0 = "vector.shuffle"(%a, %a) <{mask = array<i64: 0>}> : (vector<f32>, vector<f32>) -> vector<1x1xf32>
// CHECK: Operation does not verify: If inputs are 0-D output must be 1-D

// -----

%a, %b = "test.op"() : () -> (vector<2x3xf32>, vector<2x4xf32>)
%0 = "vector.shuffle"(%a, %b) <{mask = array<i64: 0>}> : (vector<2x3xf32>, vector<2x4xf32>) -> vector<1x3xf32>
// CHECK: Operation does not verify: Input trailing dimensions must match

// -----

%a, %b = "test.op"() : () -> (vector<2x3xf32>, vector<2x3xf32>)
%0 = "vector.shuffle"(%a, %b) <{mask = array<i64: 4>}> : (vector<2x3xf32>, vector<2x3xf32>) -> vector<1x3xf32>
// CHECK: Operation does not verify: Mask value 4 out of range [-1, 4)
