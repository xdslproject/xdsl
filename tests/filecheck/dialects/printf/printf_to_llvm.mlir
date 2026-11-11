// RUN: xdsl-opt %s -p printf-to-llvm --split-input-file | filecheck %s

builtin.module {
    "func.func"() ({
        %pi = "arith.constant"() {value = 3.14159:f32} : () -> f32
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32

        printf.print_format "Hello: {} {}\n", %pi : f32, %12 : i32
        printf.print_format "No args\n"

        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}

// CHECK:       %{{\d+}} = llvm.mlir.addressof @Hello_f_842f9d94ff2eba9703926bef3c2bc5f427db9871 : !llvm.ptr

// CHECK:       llvm.call @printf(%{{\d+}}, %{{\d+}}, %{{\d+}}) vararg(!llvm.func<void (!llvm.ptr, ...)>) : (!llvm.ptr, f64, i32) -> ()

// CHECK:       %{{\d+}} = llvm.mlir.addressof @No_args_{{\w+}} : !llvm.ptr
// CHECK-NEXT:  llvm.call @printf(%{{\d+}}) vararg(!llvm.func<void (!llvm.ptr, ...)>) : (!llvm.ptr) -> ()

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

// -----

builtin.module {
    func.func @ints(%a : i32, %b : si32, %c : ui32, %d : i64, %e : si64, %f : ui64) {
        printf.print_format "{} {} {} {} {} {}\n", %a : i32, %b : si32, %c : ui32, %d : i64, %e : si64, %f : ui64
        func.return
    }
}

// CHECK:       llvm.call @printf(%{{\d+}}, %a, %b, %c, %d, %e, %f) vararg(!llvm.func<void (!llvm.ptr, ...)>) : (!llvm.ptr, i32, si32, ui32, i64, si64, ui64) -> ()

// "%i %i %u %li %li %lu\n"
// CHECK:       llvm.mlir.global internal constant @{{\w+}}(dense<[37, 105, 32, 37, 105, 32, 37, 117, 32, 37, 108, 105, 32, 37, 108, 105, 32, 37, 108, 117, 10, 0]> : tensor<22xi8>) {addr_space = 0 : i32} : !llvm.array<22 x i8>
