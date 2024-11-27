// RUN: xdsl-opt %s -p memref-stream-infer-fill | filecheck %s

// CHECK:  builtin.module {

%Z, %zero = "test.op"() : () -> (memref<8x8xf64>, f64)
// CHECK-NEXT:   %Z, %zero = "test.op"() : () -> (memref<8x8xf64>, f64)

memref_stream.generic {
    bounds = [8, 8],
    indexing_maps = [
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%zero : f64) outs(%Z : memref<8x8xf64>) {
^bb0(%in: f64, %out: f64):
    memref_stream.yield %in : f64
}
// CHECK-NEXT:    memref_stream.fill %Z with %zero : memref<8x8xf64>

// CHECK-NEXT:  }
