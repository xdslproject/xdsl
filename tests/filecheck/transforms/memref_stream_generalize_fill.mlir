// RUN: xdsl-opt %s -p memref-stream-generalize-fill | filecheck %s

// CHECK:  builtin.module {

%Z, %zero = "test.op"() : () -> (memref<8x8xf64>, f64)
// CHECK-NEXT:   %Z, %zero = "test.op"() : () -> (memref<8x8xf64>, f64)

memref_stream.fill %Z with %zero : memref<8x8xf64>

// CHECK-NEXT:    memref_stream.generic {bounds = [8, 8], indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%zero : f64) outs(%Z : memref<8x8xf64>) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}}: f64, %{{.*}}: f64):
// CHECK-NEXT:        memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:    }

// CHECK-NEXT:  }
