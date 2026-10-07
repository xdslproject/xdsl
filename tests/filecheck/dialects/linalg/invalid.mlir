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

%A, %B, %C = "test.op"() : () -> (memref<4x6xf32>, memref<3x4xf32>, memref<4x4xf32>)
linalg.matmul ins(%A, %B : memref<4x6xf32>, memref<3x4xf32>) outs(%C : memref<4x4xf32>)

// CHECK: Operation does not verify: inferred input/output operand #1 has shape's dimension #0 to be 6, but found 3

// -----

%A, %B, %C = "test.op"() : () -> (memref<4x6xf32>, memref<6x4xf32>, memref<9x9xf32>)
linalg.matmul ins(%A, %B : memref<4x6xf32>, memref<6x4xf32>) outs(%C : memref<9x9xf32>)

// CHECK: Operation does not verify: inferred input/output operand #2 has shape's dimension #0 to be 4, but found 9

// -----

%A, %B, %C = "test.op"() : () -> (memref<4x6x2xf32>, memref<6x4xf32>, memref<4x4xf32>)
linalg.matmul ins(%A, %B : memref<4x6x2xf32>, memref<6x4xf32>) outs(%C : memref<4x4xf32>)

// CHECK: Operation does not verify: expected operand #0 of rank 3 to match the result count of indexing_map #0, 2

// -----

%input, %output = "test.op"() : () -> (tensor<4x4xf32>, memref<4x4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : tensor<4x4xf32>) outs(%output : memref<4x4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: expected to have pure tensor or buffer semantics

// -----

%input, %output = "test.op"() : () -> (memref<4x4xf32>, memref<4x4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0) -> (d0, d0)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : memref<4x4xf32>) outs(%output : memref<4x4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: expected indexing_map #1 to have 2 dim(s) to match the number of loops

// -----

%input, %output = "test.op"() : () -> (memref<8xf32>, memref<8xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>, affine_map<(d0, d1) -> (d0 + d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : memref<8xf32>) outs(%output : memref<8xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: invalid indexing maps are non-invertible: ((d0, d1) -> ((d0 + d1), (d0 + d1)))

// -----

// A result reading two loops only has to fit, and here it does not: d0 + d1
// reaches index 6 over a 4x4 output, so the input needs 7 elements and has 5.
%input, %output = "test.op"() : () -> (memref<5xf32>, memref<4x4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : memref<5xf32>) outs(%output : memref<4x4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: inferred input/output operand #0 has shape's dimension #0 to be greater than or equal to 7, but found 5

// -----

%input, %output = "test.op"() : () -> (memref<4xf32>, memref<4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0 - 1)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: unexpected result less than 0 at expression #0 in (d0) -> ((d0 + -1))

// -----

%input, %output = "test.op"() : () -> (memref<4xf32>, memref<4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
^bb0(%in: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: expected as many non-induction variable region arguments as the number of input/output operands, 2, but got 1

// -----

%input, %output = "test.op"() : () -> (memref<4xf32>, memref<4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
^bb0(%in: i32, %out: f32):
  linalg.yield %out : f32
}

// CHECK: Operation does not verify: expected type of bb argument #0 (i32) to match element or self type of the corresponding operand (f32)

// -----

%input, %output = "test.op"() : () -> (memref<4xf32>, memref<4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: expected the number of indexing_map (1) to be equal to the number of input/output operands (2)

// -----

%input, %output = "test.op"() : () -> (memref<4xf32>, memref<4xf32>)
linalg.generic {
  indexing_maps = [affine_map<(d0)[s0] -> (d0 + s0)>, affine_map<(d0)[s0] -> (d0)>],
  iterator_types = ["parallel"]
} ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}

// CHECK: Operation does not verify: unexpected symbols in indexing_map #0

// -----

%lhs, %rhs, %out = "test.op"() : () -> (memref<4xf32>, memref<5xf32>, memref<4xf32>)
linalg.add ins(%lhs, %rhs : memref<4xf32>, memref<5xf32>) outs(%out : memref<4xf32>)

// CHECK: Operation does not verify: inferred input/output operand #1 has shape's dimension #0 to be 4, but found 5

// -----

%lhs, %rhs, %out = "test.op"() : () -> (memref<4xf32>, memref<4x4xf32>, memref<4xf32>)
linalg.add ins(%lhs, %rhs : memref<4xf32>, memref<4x4xf32>) outs(%out : memref<4xf32>)

// CHECK: Operation does not verify: expected operand #1 of rank 2 to match the result count of indexing_map #1, 1
