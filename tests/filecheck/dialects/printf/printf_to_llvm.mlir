// RUN: xdsl-opt %s -p printf-to-llvm --split-input-file | filecheck %s

builtin.module {
    "func.func"() ({
        %pi = "arith.constant"() {value = 3.14159:f32} : () -> f32
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32

        printf.print_format "Hello: {} {}", %pi : f32, %12 : i32

        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}

// CHECK:       %{{\d+}} = "llvm.mlir.addressof"() <{global_name = @Hello_f_842f9d94ff2eba9703926bef3c2bc5f427db9871}> : () -> !llvm.ptr

// CHECK:       "llvm.call"(%{{\d+}}, %{{\d+}}, %{{\d+}}) <{callee = @printf{{.*}}}> : (!llvm.ptr, f64, i32) -> ()

// CHECK:       "llvm.func"() <{sym_name = "printf", function_type = !llvm.func<void (!llvm.ptr, ...)>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
// CHECK-NEXT:  }) : () -> ()

// CHECK:       "llvm.mlir.global"() <{global_type = !llvm.array<14 x i8>, sym_name = "Hello_f_842f9d94ff2eba9703926bef3c2bc5f427db9871", linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, constant, value = dense<[72, 101, 108, 108, 111, 58, 32, 37, 102, 32, 37, 105, 10, 0]> : tensor<14xi8>}> ({
// CHECK-NEXT:  }) : () -> ()

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
