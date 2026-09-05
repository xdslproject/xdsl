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

// -----

%1, %2, %3 = "test.op"() : () -> (memref<4x6xf32>, memref<3x4xf32>, memref<4x4xf32>)
linalg.matmul ins(%1, %2 : memref<4x6xf32>, memref<3x4xf32>) outs(%3 : memref<4x4xf32>)

// CHECK: Operation does not verify: dim(operand 1, 0) = 3 doesn't match the 6 given by its indexing map

// -----

%1, %2, %3 = "test.op"() : () -> (memref<4x6xf32>, memref<6x4xf32>, memref<9x9xf32>)
linalg.matmul ins(%1, %2 : memref<4x6xf32>, memref<6x4xf32>) outs(%3 : memref<9x9xf32>)

// CHECK: Operation does not verify: dim(operand 2, 0) = 9 doesn't match the 4 given by its indexing map

// -----

%1, %2, %3 = "test.op"() : () -> (memref<4x6x2xf32>, memref<6x4xf32>, memref<4x4xf32>)
linalg.matmul ins(%1, %2 : memref<4x6x2xf32>, memref<6x4xf32>) outs(%3 : memref<4x4xf32>)

// CHECK: Operation does not verify: rank(operand 0) = 3 doesn't match the number of results of its indexing map (2)

// -----

%1, %2, %3 = "test.op"() : () -> (memref<4x16xf32>, memref<8x16xf32>, memref<4x16xf32>)
linalg.add ins(%1, %2 : memref<4x16xf32>, memref<8x16xf32>) outs(%3 : memref<4x16xf32>) -> ()

// CHECK: Operation does not verify: dim(operand 1, 0) = 8 doesn't match the 4 given by its indexing map
