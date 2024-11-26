// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM

// CHECK: arm_func.return {"comment" = "this is a return instruction"}
// CHECK-ASM: bx lr # this is a return instruction
arm_func.return {"comment" = "this is a return instruction"}

// CHECK-GENERIC: "arm_func.return"() {"comment" = "this is a return instruction"} : () -> ()
