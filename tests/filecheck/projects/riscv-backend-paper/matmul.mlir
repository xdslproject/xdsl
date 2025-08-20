// RUN: xdsl-run  --args="dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf64>, dense<[[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]]> : tensor<2x3xf64>, dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<4x3xf64>" --verbose %s | filecheck %s
// RUN: xdsl-opt -p convert-linalg-to-loops %s | xdsl-run  --args="dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf64>, dense<[[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]]> : tensor<2x3xf64>, dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<4x3xf64>" --verbose | filecheck %s

func.func @main(%A: memref<4x2xf64>, %B : memref<2x3xf64>, %C : memref<4x3xf64>) -> memref<4x3xf64> {
    linalg.generic {
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
        linalg.yield %acc_new : f64
    }
    func.return %C : memref<4x3xf64>
}

// CHECK{LITERAL}: [[1.5, 2.25, 3.0], [3.0, 4.75, 6.5], [4.5, 7.25, 10.0], [6.0, 9.75, 13.5]]
