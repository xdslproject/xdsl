// RUN: xdsl-run %s | filecheck %s

builtin.module {
    "memref.global"() {"sym_name" = "a", "type" = memref<2x3xf64>, "initial_value" = dense<[[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()
    "memref.global"() {"sym_name" = "b", "type" = memref<2x3xf64>, "initial_value" = dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()

    func.func @main() {
        %A = "memref.get_global"() {"name" = @a} : () -> memref<2x3xf64>
        %B = "memref.get_global"() {"name" = @b} : () -> memref<2x3xf64>
        %zero = arith.constant 0.0 : f64
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]
        } ins(%A : memref<2x3xf64>) outs(%B : memref<2x3xf64>) {
        ^bb0(%a : f64, %b : f64):
            %res = arith.maximumf %a, %zero : f64
            linalg.yield %res : f64
        }

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index

        %v0 = memref.load %B[%c0, %c0] : memref<2x3xf64>
        %v1 = memref.load %B[%c0, %c1] : memref<2x3xf64>
        %v2 = memref.load %B[%c0, %c2] : memref<2x3xf64>
        %v3 = memref.load %B[%c1, %c0] : memref<2x3xf64>
        %v4 = memref.load %B[%c1, %c1] : memref<2x3xf64>
        %v5 = memref.load %B[%c1, %c2] : memref<2x3xf64>

        "printf.print_format"(%v0, %v1, %v2, %v3, %v4, %v5) {format_str = "[[{}, {}, {}], [{}, {}, {}]]"} : (f64, f64, f64, f64, f64, f64) -> ()
        func.return
    }
}

// CHECK{LITERAL}: [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
