// RUN: xdsl-opt %s -p print-to-printf | mlir-opt --test-lower-to-llvm | mlir-translate --mlir-to-llvmir | filecheck %s
// this tests straight to llvmir, because I feel that that's a reasonable thing to do.

builtin.module {
    "func.func"() ({
        %pi = "arith.constant"() {value = 3.14159:f32} : () -> f32
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32

        print.println "Hello: {} {}", %pi : f32, %12 : i32

        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}


// CHECK: @Hello_f_{{\w+}} = internal constant [14 x i8] c"Hello: %f %i\0A\00"

// CHECK: call void (ptr, ...) @printf(ptr @Hello_f_{{\w+}}, double {{\w+}}, i32 12)

// CHECK: declare void @printf(ptr, ...)
