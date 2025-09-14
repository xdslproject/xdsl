// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%0 = linalg.index 3 : index

// CHECK: Operation does not verify: 'linalg.index' expects parent op 'linalg.generic'

// -----

%1, %2 = "test.op"() : () -> (tensor<12x20xf32>, tensor<20xi32>)
linalg.reduce ins(%1:tensor<12x20xf32>) outs(%2:tensor<20xi32>) dimensions = [0]
(%3 : f32, %4 : f32) {
    %5 = arith.addf %3, %4 : f32
    linalg.yield %5 : f32
}

// CHECK: Operation does not verify: Reduction element types must be equal, but input is f32 and init is i32

// -----

%1, %2 = "test.op"() : () -> (tensor<12x20xf32>, tensor<10xf32>)
linalg.reduce ins(%1:tensor<12x20xf32>) outs(%2:tensor<10xf32>) dimensions = [0]
(%3 : f32, %4 : f32) {
    %5 = arith.addf %3, %4 : f32
    linalg.yield %5 : f32
}

// CHECK: Operation does not verify: Non-reduced input dimension 1 must equal output dimension 0

// -----

%1, %2 = "test.op"() : () -> (memref<12x20xf32>, memref<20xf32>)
linalg.reduce ins(%1:memref<12x20xf32>) outs(%2:memref<20xf32>) dimensions = [0, 1]
(%3 : f32, %4 : f32) {
    %5 = arith.addf %3, %4 : f32
    linalg.yield %5 : f32
}

// CHECK: Operation does not verify: Output rank must equal input rank minus number of dimensions being reduced over
