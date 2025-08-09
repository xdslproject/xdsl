// RUN: xdsl-opt -p memref-stream-unnest-out-parameters %s | filecheck %s

// CHECK:       builtin.module {

%A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)
// CHECK-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [4, 2, 3],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
^bb0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    memref_stream.yield %acc_new : f64
}

// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [4, 2, 3],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d2, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    } ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
// CHECK-NEXT:    ^bb0(%a : f64, %b : f64, %acc_old : f64):
// CHECK-NEXT:      %prod = arith.mulf %a, %b : f64
// CHECK-NEXT:      %acc_new = arith.addf %acc_old, %prod : f64
// CHECK-NEXT:      memref_stream.yield %acc_new : f64
// CHECK-NEXT:    }

%X, %Y, %Z  = "test.op"() : () -> (memref<1x1x8x8xf64>, memref<1x1x3x3xf64>, memref<1x1x6x6xf64>)
// CHECK-NEXT:    %X, %Y, %Z = "test.op"() : () -> (memref<1x1x8x8xf64>, memref<1x1x3x3xf64>, memref<1x1x6x6xf64>)

memref_stream.generic {
    bounds = [1, 1, 6, 6, 1, 3, 3],
    indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
} ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) outs(%Z : memref<1x1x6x6xf64>) {
^bb0(%x : f64, %y : f64, %acc : f64):
    %prod = arith.mulf %x, %y fastmath<fast> : f64
    %res = arith.addf %prod, %acc fastmath<fast> : f64
    memref_stream.yield %res : f64
}

// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [1, 1, 6, 6, 1, 3, 3],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, (d2 + d5), (d3 + d6))>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-NEXT:    } ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) outs(%Z : memref<1x1x6x6xf64>) {
// CHECK-NEXT:    ^bb1(%x : f64, %y : f64, %acc : f64):
// CHECK-NEXT:      %prod_1 = arith.mulf %x, %y fastmath<fast> : f64
// CHECK-NEXT:      %res = arith.addf %prod_1, %acc fastmath<fast> : f64
// CHECK-NEXT:      memref_stream.yield %res : f64
// CHECK-NEXT:    }

// CHECK-NEXT:  }
