// RUN: xdsl-opt %s -p printf-to-llvm | mlir-opt --test-lower-to-llvm  | filecheck %s
// this tests straight to llvmir to verify intended target compatibility

builtin.module {
    "func.func"() ({
        %pi = "arith.constant"() {value = 3.14159:f32} : () -> f32
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32

        printf.print_format "Hello: {} {}", %pi : f32, %12 : i32

        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}


// CHECK: llvm.call @printf(%{{\d+}}, %{{\d+}}, %{{\d+}}) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, f64, i32) -> ()

// CHECK: llvm.func @printf(!llvm.ptr, ...)

// CHECK: llvm.mlir.global internal constant @Hello_f_{{\w+}}(dense<[72, 101, 108, 108, 111, 58, 32, 37, 102, 32, 37, 105, 10, 0]> : tensor<14xi8>) {addr_space = 0 : i32} : !llvm.arr
