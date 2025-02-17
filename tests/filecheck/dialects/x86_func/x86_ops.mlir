// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t x86-asm %s | filecheck %s --check-prefix=CHECK-ASM

// CHECK:      builtin.module {
// CHECK-NEXT: x86_func.func @noarg_void() -> () {
// CHECK-NEXT:    x86_func.ret {comment = "return instruction for noarg_void"}
// CHECK-NEXT:  }
// CHECK-ASM: noarg_void:
// CHECK-ASM: ret # return instruction for noarg_void
x86_func.func @noarg_void() {
    x86_func.ret {"comment" = "return instruction for noarg_void"}
}

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT: "x86_func.func"() ({
// CHECK-GENERIC-NEXT:   "x86_func.ret"() {comment = "return instruction for noarg_void"} : () -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "noarg_void", function_type = () -> ()} : () -> ()

// CHECK:     x86_func.func @arg_i32(i32, i32) -> i32 {
// CHECK-NEXT:   x86_func.ret {comment = "return instruction for arg_i32"}
// CHECK-NEXT: }
// CHECK-ASM: arg_i32:
// CHECK-ASM: ret # return instruction for arg_i32
x86_func.func @arg_i32(i32,i32) -> (i32) {
    x86_func.ret {"comment" = "return instruction for arg_i32"}
}

// CHECK-GENERIC:      "x86_func.func"() ({
// CHECK-GENERIC-NEXT:   "x86_func.ret"() {comment = "return instruction for arg_i32"} : () -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "arg_i32", function_type = (i32, i32) -> i32} : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
