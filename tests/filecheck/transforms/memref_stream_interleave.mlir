// RUN: xdsl-opt %s -p memref-stream-interleave | filecheck %s

// CHECK:  builtin.module {

%A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>)
// CHECK-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>)

%zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %zero_float = arith.constant 0.000000e+00 : f64

memref_stream.generic {
    bounds = [3, 8, 5],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<3x5xf64>, memref<5x8xf64>) outs(%C : memref<3x8xf64>) inits(%zero_float : f64) {
^1(%a : f64, %b : f64, %c : f64):
    %prod = arith.mulf %a, %b fastmath<fast> : f64
    %res = arith.addf %prod, %c fastmath<fast> : f64
    memref_stream.yield %res : f64
}


// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [3, 2, 5, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:    } ins(%A, %B : memref<3x5xf64>, memref<5x8xf64>) outs(%C : memref<3x8xf64>) inits(%zero_float : f64) {
// CHECK-NEXT:    ^0(%a : f64, %a_1 : f64, %a_2 : f64, %a_3 : f64, %b : f64, %b_1 : f64, %b_2 : f64, %b_3 : f64, %c : f64, %c_1 : f64, %c_2 : f64, %c_3 : f64):
// CHECK-NEXT:      %0 = arith.mulf %a, %b fastmath<fast> : f64
// CHECK-NEXT:      %1 = arith.mulf %a_1, %b_1 fastmath<fast> : f64
// CHECK-NEXT:      %2 = arith.mulf %a_2, %b_2 fastmath<fast> : f64
// CHECK-NEXT:      %3 = arith.mulf %a_3, %b_3 fastmath<fast> : f64
// CHECK-NEXT:      %4 = arith.addf %0, %c fastmath<fast> : f64
// CHECK-NEXT:      %5 = arith.addf %1, %c_1 fastmath<fast> : f64
// CHECK-NEXT:      %6 = arith.addf %2, %c_2 fastmath<fast> : f64
// CHECK-NEXT:      %7 = arith.addf %3, %c_3 fastmath<fast> : f64
// CHECK-NEXT:      memref_stream.yield %4, %5, %6, %7 : f64, f64, f64, f64
// CHECK-NEXT:    }

// CHECK-NEXT:  }
