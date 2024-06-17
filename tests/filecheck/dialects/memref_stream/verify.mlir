// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>
    ],
    iterator_types = ["parallel", "reduction", "parallel"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    memref_stream.yield %acc_new : f64
}

// CHECK: Operation does not verify: Unexpected order of iterator types: ['parallel', 'reduction', 'parallel']

// -----

%A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    memref_stream.yield %acc_new : f64
}

// CHECK: Operation does not verify: The number of affine maps must match the number of operands

// -----

%A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>],
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    memref_stream.yield %acc_new : f64
}

// CHECK: Operation does not verify: Invalid number of dims in indexing map 0

// -----

%A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    memref_stream.yield %acc_new : f64
}

// CHECK: Operation does not verify: The number of dims in output indexing maps must be 3 or 2

// -----

%A, %B, %C, %D = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C, %D : memref<4x3xf64>, memref<4x3xf64>) {
^0(%a : f64, %b : f64, %acc_old0 : f64, %acc_old1 : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new0 = arith.addf %acc_old0, %prod : f64
    %acc_new1 = arith.addf %acc_old1, %prod : f64
    memref_stream.yield %acc_new0, %acc_new1 : f64, f64
}

// CHECK: Operation does not verify: The number of dims in output indexing maps must all be the same
