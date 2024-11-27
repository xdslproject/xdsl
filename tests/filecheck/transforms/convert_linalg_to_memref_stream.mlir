// RUN: xdsl-opt -p convert-linalg-to-memref-stream %s | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
%D, %E, %F = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
%G, %H, %I = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)

linalg.generic {
    indexing_maps = [
        affine_map<() -> ()>,
        affine_map<() -> ()>,
        affine_map<() -> ()>
    ],
    iterator_types = [],
    doc = "documentation string",
    library_call = "library call"
} ins(%A, %B : memref<f64>, memref<f64>) outs(%C : memref<f64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<() -> ()>,
// CHECK-NEXT:        affine_map<() -> ()>,
// CHECK-NEXT:        affine_map<() -> ()>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = [],
// CHECK-NEXT:      doc = "documentation string",
// CHECK-NEXT:      library_call = "library call"
// CHECK-NEXT:    } ins(%A, %B : memref<f64>, memref<f64>) outs(%C : memref<f64>) {
// CHECK-NEXT:    ^0(%a : f64, %b : f64, %acc_old : f64):
// CHECK-NEXT:      %prod = arith.mulf %a, %b : f64
// CHECK-NEXT:      %acc_new = arith.addf %acc_old, %prod : f64
// CHECK-NEXT:      memref_stream.yield %acc_new : f64
// CHECK-NEXT:    }

linalg.generic {
    indexing_maps = [
        affine_map<(i, j, k) -> (i, j)>,
        affine_map<(i, j, k) -> (j, k)>,
        affine_map<(i, j, k) -> (i, k)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%D, %E : memref<2x3xf64>, memref<3x4xf64>) outs(%F : memref<2x4xf64>) {
^0(%d : f64, %e : f64, %acc_old : f64):
    %prod = arith.mulf %d, %e : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [2, 3, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d1, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    } ins(%D, %E : memref<2x3xf64>, memref<3x4xf64>) outs(%F : memref<2x4xf64>) {
// CHECK-NEXT:    ^1(%d : f64, %e : f64, %acc_old_1 : f64):
// CHECK-NEXT:      %prod_1 = arith.mulf %d, %e : f64
// CHECK-NEXT:      %acc_new_1 = arith.addf %acc_old_1, %prod_1 : f64
// CHECK-NEXT:      memref_stream.yield %acc_new_1 : f64
// CHECK-NEXT:    }

linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i + j)>,
        affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
} ins(%G, %H : memref<4xf64>, memref<2xf64>) outs(%I : memref<3xf64>) {
^0(%g : f64, %h : f64, %acc_old : f64):
    %prod = arith.mulf %g, %h : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [3, 2],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1) -> ((d0 + d1))>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "reduction"]
// CHECK-NEXT:    } ins(%G, %H : memref<4xf64>, memref<2xf64>) outs(%I : memref<3xf64>) {
// CHECK-NEXT:    ^2(%g : f64, %h : f64, %acc_old_2 : f64):
// CHECK-NEXT:      %prod_2 = arith.mulf %g, %h : f64
// CHECK-NEXT:      %acc_new_2 = arith.addf %acc_old_2, %prod_2 : f64
// CHECK-NEXT:      memref_stream.yield %acc_new_2 : f64
// CHECK-NEXT:    }

// CHECK-NEXT:  }
