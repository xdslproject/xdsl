// RUN: xdsl-run %s | filecheck %s

builtin.module {
    "memref.global"() {"sym_name" = "a", "type" = memref<4x2xf64>, "initial_value" = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf64>, "sym_visibility" = "public"} : () -> ()
    "memref.global"() {"sym_name" = "b", "type" = memref<2x3xf64>, "initial_value" = dense<[[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()
    "memref.global"() {"sym_name" = "c", "type" = memref<4x3xf64>, "initial_value" = dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<4x3xf64>, "sym_visibility" = "public"} : () -> ()

    func.func @main() {
        %A = "memref.get_global"() {"name" = @a} : () -> memref<4x2xf64>
        %B = "memref.get_global"() {"name" = @b} : () -> memref<2x3xf64>
        %C = "memref.get_global"() {"name" = @c} : () -> memref<4x3xf64>
        linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d1)>,
                affine_map<(d0, d1, d2) -> (d1, d2)>,
                affine_map<(d0, d1, d2) -> (d0, d2)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
        } ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) outs(%C : memref<4x3xf64>) {
        ^0(%a : f64, %b : f64, %acc_old : f64):
            %prod = arith.mulf %a, %b : f64
            %acc_new = arith.addf %acc_old, %prod : f64
            linalg.yield %acc_new : f64
        }

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index

        %v0 = memref.load %C[%c0, %c0] : memref<4x3xf64>
        %v1 = memref.load %C[%c0, %c1] : memref<4x3xf64>
        %v2 = memref.load %C[%c0, %c2] : memref<4x3xf64>
        %v3 = memref.load %C[%c1, %c0] : memref<4x3xf64>
        %v4 = memref.load %C[%c1, %c1] : memref<4x3xf64>
        %v5 = memref.load %C[%c1, %c2] : memref<4x3xf64>
        %v6 = memref.load %C[%c2, %c0] : memref<4x3xf64>
        %v7 = memref.load %C[%c2, %c1] : memref<4x3xf64>
        %v8 = memref.load %C[%c2, %c2] : memref<4x3xf64>
        %v9 = memref.load %C[%c3, %c0] : memref<4x3xf64>
        %v10 = memref.load %C[%c3, %c1] : memref<4x3xf64>
        %v11 = memref.load %C[%c3, %c2] : memref<4x3xf64>

        "printf.print_format"(%v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7, %v8, %v9, %v10, %v11) {format_str = "[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]"} : (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> ()
        func.return
    }
}

// CHECK: [[1.5, 2.25, 3.0], [3.0, 4.75, 6.5], [4.5, 7.25, 10.0], [6.0, 9.75, 13.5]]
