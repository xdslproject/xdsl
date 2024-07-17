// RUN: xdsl-opt -p memref-stream-fold-fill %s | filecheck %s

// CHECK:       builtin.module {

%m0, %m1, %m2, %m3 = "test.op"() : () -> (memref<5xf64>, memref<5xf64>, memref<5xf64>, memref<5x3xf64>)
%s0, %s1, %s2 = "test.op"() : () -> (f64, f64, f64)

// CHECK-NEXT:    %m0, %m1, %m2, %m3 = "test.op"() : () -> (memref<5xf64>, memref<5xf64>, memref<5xf64>, memref<5x3xf64>)
// CHECK-NEXT:    %s0, %s1, %s2 = "test.op"() : () -> (f64, f64, f64)

memref_stream.fill %m0 with %s1 : memref<5xf64>
memref_stream.fill %m1 with %s2 : memref<5xf64>

// The first operand should be init with s1
// The second operand should be init with s2
// The third operand is not filled, and should not have a corresponding init

memref_stream.generic {
    bounds = [5, 3],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,  // m3
        affine_map<(d0) -> (d0)>,          // m0
        affine_map<(d0) -> (d0)>,          // m1
        affine_map<(d0) -> (d0)>           // m2
    ],
    iterator_types = ["parallel", "reduction"]
} ins(%m3 : memref<5x3xf64>) outs(%m0, %m1, %m2 : memref<5xf64>, memref<5xf64>, memref<5xf64>) {
^0(%in : f64, %out0 : f64, %out1 : f64, %out2 : f64):
    %sum0 = arith.addf %out0, %in : f64
    %sum1 = arith.addf %out1, %in : f64
    %sum2 = arith.addf %out2, %in : f64
    memref_stream.yield %sum0, %sum1, %sum2 : f64, f64, f64
}
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [5, 3],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0) -> (d0)>,
// CHECK-NEXT:        affine_map<(d0) -> (d0)>,
// CHECK-NEXT:        affine_map<(d0) -> (d0)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "reduction"]
// CHECK-NEXT:    } ins(%m3 : memref<5x3xf64>) outs(%m0, %m1, %m2 : memref<5xf64>, memref<5xf64>, memref<5xf64>) inits(%s1 : f64, %s2 : f64, None) {
// CHECK-NEXT:    ^0(%in : f64, %out0 : f64, %out1 : f64, %out2 : f64):
// CHECK-NEXT:      %sum0 = arith.addf %out0, %in : f64
// CHECK-NEXT:      %sum1 = arith.addf %out1, %in : f64
// CHECK-NEXT:      %sum2 = arith.addf %out2, %in : f64
// CHECK-NEXT:      memref_stream.yield %sum0, %sum1, %sum2 : f64, f64, f64
// CHECK-NEXT:    }


// Unnested parameters should not be folded

memref_stream.fill %m0 with %s1 : memref<5xf64>
memref_stream.fill %m1 with %s2 : memref<5xf64>
memref_stream.generic {
    bounds = [5, 3],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,  // m3
        affine_map<(d0, d1) -> (d0)>,      // m0
        affine_map<(d0, d1) -> (d0)>,      // m1
        affine_map<(d0, d1) -> (d0)>       // m2
    ],
    iterator_types = ["parallel", "reduction"]
} ins(%m3 : memref<5x3xf64>) outs(%m0, %m1, %m2 : memref<5xf64>, memref<5xf64>, memref<5xf64>) {
^0(%in : f64, %out0 : f64, %out1 : f64, %out2 : f64):
    %sum0 = arith.addf %out0, %in : f64
    %sum1 = arith.addf %out1, %in : f64
    %sum2 = arith.addf %out2, %in : f64
    memref_stream.yield %sum0, %sum1, %sum2 : f64, f64, f64
}
// CHECK-NEXT:    memref_stream.fill %m0 with %s1 : memref<5xf64>
// CHECK-NEXT:    memref_stream.fill %m1 with %s2 : memref<5xf64>
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [5, 3],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "reduction"]
// CHECK-NEXT:    } ins(%m3 : memref<5x3xf64>) outs(%m0, %m1, %m2 : memref<5xf64>, memref<5xf64>, memref<5xf64>) {
// CHECK-NEXT:    ^1(%in_1 : f64, %out0_1 : f64, %out1_1 : f64, %out2_1 : f64):
// CHECK-NEXT:      %sum0_1 = arith.addf %out0_1, %in_1 : f64
// CHECK-NEXT:      %sum1_1 = arith.addf %out1_1, %in_1 : f64
// CHECK-NEXT:      %sum2_1 = arith.addf %out2_1, %in_1 : f64
// CHECK-NEXT:      memref_stream.yield %sum0_1, %sum1_1, %sum2_1 : f64, f64, f64
// CHECK-NEXT:    }


// Two consecutive fills should be folded
memref_stream.fill %m0 with %s0 : memref<5xf64>
memref_stream.fill %m0 with %s1 : memref<5xf64>
// CHECK-NEXT:    memref_stream.fill %m0 with %s1 : memref<5xf64>

// CHECK-NEXT:  }
