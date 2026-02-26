// RUN: xdsl-run --verbose %s | filecheck %s

builtin.module {
    func.func @main() -> (tensor<?xi32>, tensor<4xi32>, tensor<4xi1>) {
        %0 = arith.constant 0 : index
        %1 = arith.constant 1 : index
        %2 = arith.constant 10 : i32
        %x_length = arith.index_cast %2 : i32 to index
        %x_uninit = tensor.empty(%x_length) : tensor<?xi32>
        %x = scf.for %i = %0 to %x_length step %1 iter_args(%3 = %x_uninit) -> (tensor<?xi32>) {
            %4 = arith.index_cast %i : index to i32
            %x_modified = tensor.insert %4 into %3[%i] : tensor<?xi32>
            scf.yield %x_modified : tensor<?xi32>
        }

        %c11 = arith.constant 11 : i32
        %5 = arith.constant 4 : i32
        %y_length = arith.index_cast %5 : i32 to index
        %y_uninit = tensor.empty() : tensor<4xi32>
        %y = scf.for %i_1 = %0 to %y_length step %1 iter_args(%6 = %y_uninit) -> (tensor<4xi32>) {
            %7 = arith.index_cast %i_1 : index to i32
            %8 = arith.addi %7, %c11 : i32
            %y_modified = tensor.insert %8 into %6[%i_1] : tensor<4xi32>
            scf.yield %y_modified : tensor<4xi32>
        }

        %c1 = arith.constant 1 : i32
        %y_bool_uninit = tensor.empty() : tensor<4xi1>
        %y_bool = scf.for %i_1 = %0 to %y_length step %1 iter_args(%6 = %y_bool_uninit) -> (tensor<4xi1>) {
            %7 = arith.index_cast %i_1 : index to i32
            %8 = arith.andi %7, %c1 : i32
            %even = arith.cmpi eq, %8, %c1 : i32
            %y_modified = tensor.insert %even into %6[%i_1] : tensor<4xi1>
            scf.yield %y_modified : tensor<4xi1>
        }

        %9 = tensor.dim %y, %0 : tensor<4xi32>
        printf.print_format "enumerating:\n"
        scf.for %i_2 = %0 to %9 step %1 {
            %10 = tensor.extract %y[%i_2] : tensor<4xi32>
            printf.print_format "{}\n", %10 : i32
        }
        func.return %x, %y, %y_bool : tensor<?xi32>, tensor<4xi32>, tensor<4xi1>
    }
}

// CHECK:      enumerating:
// CHECK-NEXT: 11
// CHECK-NEXT: 12
// CHECK-NEXT: 13
// CHECK-NEXT: 14
// CHECK-NEXT: result: (
// CHECK-NEXT:     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
// CHECK-NEXT:     [11, 12, 13, 14],
// CHECK-NEXT:     [0, 1, 0, 1]
// CHECK-NEXT: )
