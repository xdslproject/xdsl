// RUN: xdsl-run %s | filecheck %s

builtin.module {
    "memref.global"() {"sym_name" = "a", "type" = memref<2x3xf64>, "initial_value" = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()
    "memref.global"() {"sym_name" = "b", "type" = memref<2x3xf64>, "initial_value" = dense<[[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()
    "memref.global"() {"sym_name" = "c", "type" = memref<2x3xf64>, "initial_value" = dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<2x3xf64>, "sym_visibility" = "public"} : () -> ()

    func.func @main() {
        %A = "memref.get_global"() {"name" = @a} : () -> memref<2x3xf64>
        %B = "memref.get_global"() {"name" = @b} : () -> memref<2x3xf64>
        %C = "memref.get_global"() {"name" = @c} : () -> memref<2x3xf64>
        "linalg.generic"(%A, %B, %C) ({
        ^bb0(%a: f64, %b: f64, %c: f64):
            %sum = arith.addf %a, %b : f64
            linalg.yield %sum : f64
        }) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>} : (memref<2x3xf64>, memref<2x3xf64>, memref<2x3xf64>) -> ()
        printf.print_format "{}", %C : memref<2x3xf64>
        func.return
    }
}

// CHECK: [[1.0, 2.25, 3.5], [4.75, 6.0, 7.25]]
