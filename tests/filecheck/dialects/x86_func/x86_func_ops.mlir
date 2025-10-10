// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t x86-asm %s | filecheck %s --check-prefix=CHECK-ASM

// CHECK:       x86_func.func @noarg_void() {
// CHECK-NEXT:    x86_func.ret {comment = "this is a return instruction"}
// CHECK-NEXT:  }
// CHECK-ASM: noarg_void:
// CHECK-ASM: ret # this is a return instruction
x86_func.func @noarg_void() {
    x86_func.ret {"comment" = "this is a return instruction"}
}

// CHECK-GENERIC:       "x86_func.func"() ({
// CHECK-GENERIC-NEXT:    "x86_func.ret"() {comment = "this is a return instruction"} : () -> ()
// CHECK-GENERIC-NEXT:  }) {sym_name = "noarg_void", function_type = () -> ()} : () -> ()
