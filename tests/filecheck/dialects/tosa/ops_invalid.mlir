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

// -----

%simple = "test.op"() : () -> tensor<1x2x3x4xi32>
%float = "test.op"() : () -> tensor<1x2x3x4xf32>
%0 = tosa.sub %simple, %float : (tensor<1x2x3x4xi32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: attribute i32 expected from variable 'T', but got f32

// -----


%simple = "test.op"() : () -> tensor<1x2x3x4xi32>
%flat = "test.op"() : () -> tensor<1x1x1x1xi32>
%1 = tosa.sub %simple, %flat : (tensor<1x2x3x4xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
// CHECK: 'tosa.sub' Operand and result tensor shapes are not compatible

// -----


%simple = "test.op"() : () -> tensor<1x2x3x4xi32>
%float = "test.op"() : () -> tensor<1x2x3x4xf32>
%shift = "test.op"() : () -> tensor<1xi8>
%0 = tosa.mul %simple, %float, %shift : (tensor<1x2x3x4xi32>, tensor<1x2x3x4xf32>, tensor<1xi8>) -> tensor<1x2x3x4xf32>
// CHECK: attribute i32 expected from variable 'T', but got f32

// -----


%simple = "test.op"() : () -> tensor<1x2x3x4xi32>
%flat = "test.op"() : () -> tensor<1x1x1x1xi32>
%shift = "test.op"() : () -> tensor<1xi8>
%1 = tosa.mul %simple, %flat, %shift : (tensor<1x2x3x4xi32>, tensor<1x1x1x1xi32>, tensor<1xi8>) -> tensor<1x1x1x1xi32>
// CHECK: 'tosa.mul' Operand and result tensor shapes are not compatible

// -----


%not_3d = "test.op"() : () -> tensor<5x5xf32>
%zero_p = "test.op"() : () -> tensor<f32>
%0 = tosa.matmul %not_3d, %not_3d, %zero_p, %zero_p : (tensor<5x5xf32>, tensor<5x5xf32>, tensor<f32>, tensor<f32>) -> tensor<5x5xf32>
// CHECK: 'tosa.matmul' Expected operand tensors of rank 3

// -----


%not_dim_1 = "test.op"() : () -> tensor<2x5x5xf32>
%zero_p = "test.op"() : () -> tensor<f32>
%0 = tosa.matmul %not_dim_1, %not_dim_1, %zero_p, %zero_p : (tensor<2x5x5xf32>, tensor<2x5x5xf32>, tensor<f32>, tensor<f32>) -> tensor<2x5x5xf32>
// CHECK: 'tosa.matmul' Expected leading dimension of input tensors to be 1

// -----


%a = "test.op"() : () -> tensor<1x13x16xf32>
%b = "test.op"() : () -> tensor<1x23x16xf32>
%zero_p = "test.op"() : () -> tensor<f32>
%0 = tosa.matmul %a, %b, %zero_p, %zero_p : (tensor<1x13x16xf32>, tensor<1x23x16xf32>, tensor<f32>, tensor<f32>) -> tensor<1x16x23xf32>
// CHECK: 'tosa.matmul' Incompatible shapes for performing matrix multiplication

// -----

%a = "test.op"() : () -> tensor<1x13x16xf32>
%b = "test.op"() : () -> tensor<1x16x23xf32>
%zero_p = "test.op"() : () -> tensor<1x1xf32>
%0 = tosa.matmul %a, %b, %zero_p, %zero_p : (tensor<1x13x16xf32>, tensor<1x16x23xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x13x23xf32>
// CHECK: 'tosa.matmul' Expected zero-point operands to be unranked or scalar tensors
