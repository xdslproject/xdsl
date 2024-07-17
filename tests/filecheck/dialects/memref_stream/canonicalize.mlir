// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// Remove unused argument

// CHECK:  %E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)
%E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)

// CHECK-NEXT:  %I = "test.op"() : () -> memref<4x3xf64>
%I = "test.op"() : () -> memref<4x3xf64>

// Don't remove when there are uses
// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [4, 2, 3],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d2, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:  } ins(%{{\S*}}, %{{\S*}} : memref<4x2xf64>, memref<2x3xf64>) outs(%{{\S*}}, %{{\S*}} : memref<4x3xf64>, memref<4x3xf64>) inits(%{{\S*}} : f64, None) {
// CHECK-NEXT:  ^{{\S*}}(%{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64):
// CHECK-NEXT:    %{{\S*}} = arith.mulf %{{\S*}}, %{{\S*}} : f64
// CHECK-NEXT:    %{{\S*}} = arith.addf %{{\S*}}, %{{\S*}} : f64
// CHECK-NEXT:    %{{\S*}} = arith.addf %{{\S*}}, %{{\S*}} : f64
// CHECK-NEXT:    linalg.yield %{{\S*}}, %{{\S*}} : f64, f64
// CHECK-NEXT:  }
memref_stream.generic {
    bounds = [4, 2, 3],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%E, %F : memref<4x2xf64>, memref<2x3xf64>) outs(%G, %I : memref<4x3xf64>, memref<4x3xf64>) inits(%H : f64, None) {
^0(%e : f64, %f : f64, %acc_old_0 : f64, %acc_old_1 : f64):
    %prod = arith.mulf %e, %f : f64
    %acc_new_0 = arith.addf %acc_old_0, %prod : f64
    %acc_new_1 = arith.addf %acc_old_1, %prod : f64
    linalg.yield %acc_new_0, %acc_new_1 : f64, f64
}


// Do remove when there are no uses
// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [4, 2, 3],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:  } ins(%{{\S*}} : memref<4x2xf64>) outs(%{{\S*}}, %{{\S*}} : memref<4x3xf64>, memref<4x3xf64>) inits(%H : f64, None) {
// CHECK-NEXT:  ^{{\S*}}(%{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64):
// CHECK-NEXT:    linalg.yield %{{\S*}}, %{{\S*}} : f64, f64
// CHECK-NEXT:  }
memref_stream.generic {
    bounds = [4, 2, 3],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%E, %F : memref<4x2xf64>, memref<2x3xf64>) outs(%G, %I : memref<4x3xf64>, memref<4x3xf64>) inits(%H : f64, None) {
^0(%e : f64, %f : f64, %acc_old_0 : f64, %acc_old_1 : f64):
    linalg.yield %e, %acc_old_1 : f64, f64
}

// Don't remove when interleaved
// CHECK-NEXT:  func.func @interleaved_no_init(%{{\S*}} : memref<3x5xf64>, %{{\S*}} : memref<5x8xf64>, %{{\S*}} : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [3, 2, 5, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:    } ins(%{{\S*}}, %{{\S*}} : memref<3x5xf64>, memref<5x8xf64>) outs(%{{\S*}} : memref<3x8xf64>) {
// CHECK-NEXT:    ^{{\S*}}(%{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64, %{{\S*}} : f64):
// CHECK-NEXT:      %{{\S*}} = arith.addf %{{\S*}}, %{{\S*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{\S*}} = arith.addf %{{\S*}}, %{{\S*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{\S*}} = arith.addf %{{\S*}}, %{{\S*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{\S*}} = arith.addf %{{\S*}}, %{{\S*}} fastmath<fast> : f64
// CHECK-NEXT:      memref_stream.yield %{{\S*}}, %{{\S*}}, %{{\S*}}, %{{\S*}} : f64, f64, f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %{{\S*}} : memref<3x8xf64>
// CHECK-NEXT:  }
func.func @interleaved_no_init(%A0 : memref<3x5xf64>, %B0 : memref<5x8xf64>, %C0 : memref<3x8xf64>) -> memref<3x8xf64> {
    memref_stream.generic {
        bounds = [3, 2, 5, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
            affine_map<(d0, d1, d2, d3) -> (d2, d1 * 4 + d3)>,
            affine_map<(d0, d1, d3) -> (d0, d1 * 4 + d3)>
        ],
        iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
    } ins(%A0, %B0 : memref<3x5xf64>, memref<5x8xf64>) outs(%C0 : memref<3x8xf64>) {
    ^1(
        %a0 : f64, %a1 : f64, %a2 : f64, %a3 : f64,
        %b0 : f64, %b1 : f64, %b2 : f64, %b3 : f64,
        %c0 : f64, %c1 : f64, %c2 : f64, %c3 : f64
    ):
        %res0 = arith.addf %a0, %c0 fastmath<fast> : f64
        %res1 = arith.addf %a1, %c1 fastmath<fast> : f64
        %res2 = arith.addf %a2, %c2 fastmath<fast> : f64
        %res3 = arith.addf %a3, %c3 fastmath<fast> : f64

        memref_stream.yield %res0, %res1, %res2, %res3 : f64, f64, f64, f64
    }
    func.return %C0 : memref<3x8xf64>
}
