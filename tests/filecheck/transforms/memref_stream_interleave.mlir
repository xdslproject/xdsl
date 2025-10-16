// RUN: xdsl-opt %s -p memref-stream-interleave | filecheck %s
// RUN: xdsl-opt %s -p memref-stream-interleave{pipeline-depth=2} | filecheck %s --check-prefix=DEPTH-2
// RUN: xdsl-opt %s -p memref-stream-interleave{pipeline-depth=3} | filecheck %s --check-prefix=DEPTH-3
// RUN: xdsl-opt %s -p memref-stream-interleave{pipeline-depth=5} | filecheck %s --check-prefix=DEPTH-5
// RUN: xdsl-opt %s -p memref-stream-interleave{op-index=3\ iterator-index=1\ unroll-factor=4} | filecheck %s
// RUN: xdsl-opt %s -p memref-stream-interleave{op_index=3\ iterator_index=0\ unroll_factor=3} | filecheck %s --check-prefix=MANUAL-0

// CHECK:  builtin.module {

%A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x44xf64>, memref<3x44xf64>)
// CHECK-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x44xf64>, memref<3x44xf64>)

%zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %zero_float = arith.constant 0.000000e+00 : f64

memref_stream.generic {
    bounds = [3, 44, 5],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<3x5xf64>, memref<5x44xf64>) outs(%C : memref<3x44xf64>) inits(%zero_float : f64) {
^bb1(%a : f64, %b : f64, %c : f64):
    %prod = arith.mulf %a, %b fastmath<fast> : f64
    %res = arith.addf %prod, %c fastmath<fast> : f64
    memref_stream.yield %res : f64
}


// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [3, 11, 5, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:    } ins(%A, %B : memref<3x5xf64>, memref<5x44xf64>) outs(%C : memref<3x44xf64>) inits(%zero_float : f64) {
// CHECK-NEXT:    ^bb0(%a : f64, %a_1 : f64, %a_2 : f64, %a_3 : f64, %b : f64, %b_1 : f64, %b_2 : f64, %b_3 : f64, %c : f64, %c_1 : f64, %c_2 : f64, %c_3 : f64):
// CHECK-NEXT:      %prod = arith.mulf %a, %b fastmath<fast> : f64
// CHECK-NEXT:      %prod_1 = arith.mulf %a_1, %b_1 fastmath<fast> : f64
// CHECK-NEXT:      %prod_2 = arith.mulf %a_2, %b_2 fastmath<fast> : f64
// CHECK-NEXT:      %prod_3 = arith.mulf %a_3, %b_3 fastmath<fast> : f64
// CHECK-NEXT:      %res = arith.addf %prod, %c fastmath<fast> : f64
// CHECK-NEXT:      %res_1 = arith.addf %prod_1, %c_1 fastmath<fast> : f64
// CHECK-NEXT:      %res_2 = arith.addf %prod_2, %c_2 fastmath<fast> : f64
// CHECK-NEXT:      %res_3 = arith.addf %prod_3, %c_3 fastmath<fast> : f64
// CHECK-NEXT:      memref_stream.yield %res, %res_1, %res_2, %res_3 : f64, f64, f64, f64
// CHECK-NEXT:    }

// CHECK-NEXT:  }

// 44 is divisible by 2, so it should factor by 2
// DEPTH-2:       builtin.module {
// DEPTH-2-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x44xf64>, memref<3x44xf64>)
// DEPTH-2-NEXT:    %zero_float = arith.constant 0.000000e+00 : f64
// DEPTH-2-NEXT:    memref_stream.generic {
// DEPTH-2-NEXT:      bounds = [3, 22, 5, 2],
// DEPTH-2-NEXT:      indexing_maps = [
// DEPTH-2-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// DEPTH-2-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 2) + d3))>,
// DEPTH-2-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 2) + d2))>
// DEPTH-2-NEXT:      ],
// DEPTH-2-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// DEPTH-2-NEXT:    } ins(%A, %B : memref<3x5xf64>, memref<5x44xf64>) outs(%C : memref<3x44xf64>) inits(%zero_float : f64) {
// DEPTH-2-NEXT:    ^{{.*}}(%a : f64, %a_1 : f64, %b : f64, %b_1 : f64, %c : f64, %c_1 : f64):
// DEPTH-2-NEXT:      %prod = arith.mulf %a, %b fastmath<fast> : f64
// DEPTH-2-NEXT:      %prod_1 = arith.mulf %a_1, %b_1 fastmath<fast> : f64
// DEPTH-2-NEXT:      %res = arith.addf %prod, %c fastmath<fast> : f64
// DEPTH-2-NEXT:      %res_1 = arith.addf %prod_1, %c_1 fastmath<fast> : f64
// DEPTH-2-NEXT:      memref_stream.yield %res, %res_1 : f64, f64
// DEPTH-2-NEXT:    }
// DEPTH-2-NEXT:  }

// 44 is not divisible by 3, so it should factor by the next largest number that divides 44
// DEPTH-3:       builtin.module {
// DEPTH-3-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x44xf64>, memref<3x44xf64>)
// DEPTH-3-NEXT:    %zero_float = arith.constant 0.000000e+00 : f64
// DEPTH-3-NEXT:    memref_stream.generic {
// DEPTH-3-NEXT:      bounds = [3, 11, 5, 4],
// DEPTH-3-NEXT:      indexing_maps = [
// DEPTH-3-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// DEPTH-3-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// DEPTH-3-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// DEPTH-3-NEXT:      ],
// DEPTH-3-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// DEPTH-3-NEXT:    } ins(%A, %B : memref<3x5xf64>, memref<5x44xf64>) outs(%C : memref<3x44xf64>) inits(%zero_float : f64) {
// DEPTH-3-NEXT:    ^{{.*}}(%a : f64, %a_1 : f64, %a_2 : f64, %a_3 : f64, %b : f64, %b_1 : f64, %b_2 : f64, %b_3 : f64, %c : f64, %c_1 : f64, %c_2 : f64, %c_3 : f64):
// DEPTH-3-NEXT:      %prod = arith.mulf %a, %b fastmath<fast> : f64
// DEPTH-3-NEXT:      %prod_1 = arith.mulf %a_1, %b_1 fastmath<fast> : f64
// DEPTH-3-NEXT:      %prod_2 = arith.mulf %a_2, %b_2 fastmath<fast> : f64
// DEPTH-3-NEXT:      %prod_3 = arith.mulf %a_3, %b_3 fastmath<fast> : f64
// DEPTH-3-NEXT:      %res = arith.addf %prod, %c fastmath<fast> : f64
// DEPTH-3-NEXT:      %res_1 = arith.addf %prod_1, %c_1 fastmath<fast> : f64
// DEPTH-3-NEXT:      %res_2 = arith.addf %prod_2, %c_2 fastmath<fast> : f64
// DEPTH-3-NEXT:      %res_3 = arith.addf %prod_3, %c_3 fastmath<fast> : f64
// DEPTH-3-NEXT:      memref_stream.yield %res, %res_1, %res_2, %res_3 : f64, f64, f64, f64
// DEPTH-3-NEXT:    }
// DEPTH-3-NEXT:  }

// 44 is not divisible by 5, nor any numbers in 6-9, so it should factor by the next
// smallest number that divides 44, which is 4

// DEPTH-5:       builtin.module {
// DEPTH-5-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x44xf64>, memref<3x44xf64>)
// DEPTH-5-NEXT:    %zero_float = arith.constant 0.000000e+00 : f64
// DEPTH-5-NEXT:    memref_stream.generic {
// DEPTH-5-NEXT:      bounds = [3, 11, 5, 4],
// DEPTH-5-NEXT:      indexing_maps = [
// DEPTH-5-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// DEPTH-5-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// DEPTH-5-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// DEPTH-5-NEXT:      ],
// DEPTH-5-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// DEPTH-5-NEXT:    } ins(%A, %B : memref<3x5xf64>, memref<5x44xf64>) outs(%C : memref<3x44xf64>) inits(%zero_float : f64) {
// DEPTH-5-NEXT:    ^{{.*}}(%a : f64, %a_1 : f64, %a_2 : f64, %a_3 : f64, %b : f64, %b_1 : f64, %b_2 : f64, %b_3 : f64, %c : f64, %c_1 : f64, %c_2 : f64, %c_3 : f64):
// DEPTH-5-NEXT:      %prod = arith.mulf %a, %b fastmath<fast> : f64
// DEPTH-5-NEXT:      %prod_1 = arith.mulf %a_1, %b_1 fastmath<fast> : f64
// DEPTH-5-NEXT:      %prod_2 = arith.mulf %a_2, %b_2 fastmath<fast> : f64
// DEPTH-5-NEXT:      %prod_3 = arith.mulf %a_3, %b_3 fastmath<fast> : f64
// DEPTH-5-NEXT:      %res = arith.addf %prod, %c fastmath<fast> : f64
// DEPTH-5-NEXT:      %res_1 = arith.addf %prod_1, %c_1 fastmath<fast> : f64
// DEPTH-5-NEXT:      %res_2 = arith.addf %prod_2, %c_2 fastmath<fast> : f64
// DEPTH-5-NEXT:      %res_3 = arith.addf %prod_3, %c_3 fastmath<fast> : f64
// DEPTH-5-NEXT:      memref_stream.yield %res, %res_1, %res_2, %res_3 : f64, f64, f64, f64
// DEPTH-5-NEXT:    }
// DEPTH-5-NEXT:  }

// MANUAL-0:       builtin.module {
// MANUAL-0-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<3x5xf64>, memref<5x44xf64>, memref<3x44xf64>)
// MANUAL-0-NEXT:    %zero_float = arith.constant 0.000000e+00 : f64
// MANUAL-0-NEXT:    memref_stream.generic {
// MANUAL-0-NEXT:      bounds = [1, 44, 5, 3],
// MANUAL-0-NEXT:      indexing_maps = [
// MANUAL-0-NEXT:        affine_map<(d0, d1, d2, d3) -> (((d0 * 3) + d3), d2)>,
// MANUAL-0-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
// MANUAL-0-NEXT:        affine_map<(d0, d1, d2) -> (((d0 * 3) + d2), d1)>
// MANUAL-0-NEXT:      ],
// MANUAL-0-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// MANUAL-0-NEXT:    } ins(%A, %B : memref<3x5xf64>, memref<5x44xf64>) outs(%C : memref<3x44xf64>) inits(%zero_float : f64) {
// MANUAL-0-NEXT:    ^bb0(%a : f64, %a_1 : f64, %a_2 : f64, %b : f64, %b_1 : f64, %b_2 : f64, %c : f64, %c_1 : f64, %c_2 : f64):
// MANUAL-0-NEXT:      %prod = arith.mulf %a, %b fastmath<fast> : f64
// MANUAL-0-NEXT:      %prod_1 = arith.mulf %a_1, %b_1 fastmath<fast> : f64
// MANUAL-0-NEXT:      %prod_2 = arith.mulf %a_2, %b_2 fastmath<fast> : f64
// MANUAL-0-NEXT:      %res = arith.addf %prod, %c fastmath<fast> : f64
// MANUAL-0-NEXT:      %res_1 = arith.addf %prod_1, %c_1 fastmath<fast> : f64
// MANUAL-0-NEXT:      %res_2 = arith.addf %prod_2, %c_2 fastmath<fast> : f64
// MANUAL-0-NEXT:      memref_stream.yield %res, %res_1, %res_2 : f64, f64, f64
// MANUAL-0-NEXT:    }
// MANUAL-0-NEXT:  }
