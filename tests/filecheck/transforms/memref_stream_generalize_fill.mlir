// RUN: xdsl-opt %s -p memref-stream-generalize-fill | filecheck %s

// CHECK:  builtin.module {

%Z, %zero = "test.op"() : () -> (memref<8x8xf64>, f64)
// CHECK-NEXT:   %Z, %zero = "test.op"() : () -> (memref<8x8xf64>, f64)

memref_stream.fill %Z with %zero : memref<8x8xf64>

// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [8, 8],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1) -> ()>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:    } ins(%zero : f64) outs(%Z : memref<8x8xf64>) {
// CHECK-NEXT:    ^bb0(%{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:      memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:    }

// CHECK-NEXT:  }
