// RUN: xdsl-run %s | filecheck %s

builtin.module {
    "memref.global"() {"sym_name" = "a", "type" = memref<2x3xf64>, "initial_value" = dense<[[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()
    "memref.global"() {"sym_name" = "b", "type" = memref<2x3xf64>, "initial_value" = dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()

    func.func @main() {
        %A = "memref.get_global"() {"name" = @a} : () -> memref<2x3xf64>
        printf.print_format "{}", %A : memref<2x3xf64>
        %B = "memref.get_global"() {"name" = @b} : () -> memref<2x3xf64>
        %zero = arith.constant 0.0 : f64
        "linalg.generic"(%A, %B) ({
        ^bb0(%a: f64, %b: f64):
            %res = arith.maxf %a, %zero : f64
            linalg.yield %res : f64
        }) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>} : (memref<2x3xf64>, memref<2x3xf64>) -> ()
        printf.print_format "{}", %B : memref<2x3xf64>
        func.return
    }
}

// CHECK: [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
