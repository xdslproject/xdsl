// RUN: xdsl-opt -p memref-stream-unnest-out-parameters %s | filecheck %s


%A, %B, %C = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)

memref_stream.generic {
    bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    memref_stream.yield %acc_new : f64
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>)
// CHECK-NEXT:    memref_stream.generic {bounds = [#builtin.int<4>, #builtin.int<2>, #builtin.int<3>], indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) outs(%{{.*}} : memref<4x3xf64>) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:    }
// CHECK-NEXT:  }
