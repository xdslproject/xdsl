// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM

// CHECK:       arm_func.func @noarg_void() {
// CHECK-NEXT:    arm_func.return {comment = "this is a return instruction"}
// CHECK-NEXT:  }
// CHECK-ASM: noarg_void:
// CHECK-ASM: ret # this is a return instruction
arm_func.func @noarg_void() {
    arm_func.return {"comment" = "this is a return instruction"}
}

// CHECK-GENERIC:       "arm_func.func"() ({
// CHECK-GENERIC-NEXT:    "arm_func.return"() {comment = "this is a return instruction"} : () -> ()
// CHECK-GENERIC-NEXT:  }) {sym_name = "noarg_void", function_type = () -> ()} : () -> ()
