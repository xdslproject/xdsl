// RUN: xdsl-opt %s -p printf-to-llvm --split-input-file | filecheck %s

builtin.module {
    "func.func"() ({
        %pi = "arith.constant"() {value = 3.14159:f32} : () -> f32
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32

        printf.print_format "Hello: {} {}\n", %pi : f32, %12 : i32

        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}

// CHECK:       %{{\d+}} = llvm.mlir.addressof @Hello_f_842f9d94ff2eba9703926bef3c2bc5f427db9871 : !llvm.ptr

// CHECK:       llvm.call @printf(%{{\d+}}, %{{\d+}}, %{{\d+}}){{.*}} : (!llvm.ptr, f64, i32) -> ()

// CHECK:       llvm.func @printf(!llvm.ptr, ...)

// CHECK:       llvm.mlir.global internal constant @Hello_f_842f9d94ff2eba9703926bef3c2bc5f427db9871(dense<[72, 101, 108, 108, 111, 58, 32, 37, 102, 32, 37, 105, 10, 0]> : tensor<14xi8>) {addr_space = 0 : i32} : !llvm.array<14 x i8>

// -----

builtin.module {
    "func.func"() ({
        %pi = "arith.constant"() {value = 3.14159:f32} : () -> f32
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32
        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}
// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @main() {
// CHECK-NEXT:      %pi = arith.constant 3.141590e+00 : f32
// CHECK-NEXT:      %0 = arith.constant 12 : i32
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
