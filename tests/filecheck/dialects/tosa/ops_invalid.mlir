// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect | filecheck %s

%simple = "test.op"() : () -> tensor<1x2x3x4xi32>
%float = "test.op"() : () -> tensor<1x2x3x4xf32>
%0 = tosa.add %simple, %float : (tensor<1x2x3x4xi32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: attribute i32 expected from variable 'T', but got f32 

// -----


%simple = "test.op"() : () -> tensor<1x2x3x4xi32>
%flat = "test.op"() : () -> tensor<1x1x1x1xi32>
%1 = tosa.add %simple, %flat : (tensor<1x2x3x4xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
// CHECK: 'tosa.add' Operand and result tensor shapes are not compatible 

