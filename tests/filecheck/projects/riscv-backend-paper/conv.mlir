// RUN: xdsl-run %s | filecheck %s
// RUN: xdsl-opt -p convert-linalg-to-loops %s | xdsl-run | filecheck %s

module {
    "memref.global"() {
        "sym_name" = "a",
        "type" = memref<1x1x3x3xf64>,
        "initial_value" = dense<[[[[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]]]]> : tensor<1x1x3x3xf64>,
        "sym_visibility" = "public"
    } : () -> ()
    "memref.global"() {
        "sym_name" = "b",
        "type" = memref<1x1x2x2xf64>,
        "initial_value" = dense<[[[[0.0, 0.25],
                                   [0.5, 0.75]]]]> : tensor<1x1x2x2xf64>,
        "sym_visibility" = "public"
    } : () -> ()
    "memref.global"() {
        "sym_name" = "c",
        "type" = memref<1x1x2x2xf64>,
        "initial_value" = dense<[[[[0.0, 0.0],
                                   [0.0, 0.0]]]]> : tensor<1x1x2x2xf64>,
        "sym_visibility" = "public"
    } : () -> ()

    func.func public @main() {
        %A = "memref.get_global"() {"name" = @a} : () -> memref<1x1x3x3xf64>
        %B = "memref.get_global"() {"name" = @b} : () -> memref<1x1x2x2xf64>
        %C = "memref.get_global"() {"name" = @c} : () -> memref<1x1x2x2xf64>

        linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
                affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
                affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
            ],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
        } ins(%A, %B : memref<1x1x3x3xf64>, memref<1x1x2x2xf64>) outs(%C : memref<1x1x2x2xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
            %0 = arith.mulf %in, %in_0 : f64
            %1 = arith.addf %out, %0 : f64
            linalg.yield %1 : f64
        }

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        %v0 = memref.load %C[%c0, %c0, %c0, %c0] : memref<1x1x2x2xf64>
        %v1 = memref.load %C[%c0, %c0, %c0, %c1] : memref<1x1x2x2xf64>
        %v2 = memref.load %C[%c0, %c0, %c1, %c0] : memref<1x1x2x2xf64>
        %v3 = memref.load %C[%c0, %c0, %c1, %c1] : memref<1x1x2x2xf64>

        "printf.print_format"(%v0, %v1, %v2, %v3) {format_str = "[[[[{}, {}], [{}, {}]]]]"} : (f64, f64, f64, f64) -> ()
        return
    }
}

// CHECK{LITERAL}: [[6.25, 7.75], [10.75, 12.25]]
