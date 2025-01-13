// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM

module {
  // CHECK: %x1 = arm.get_register : !arm.reg<x1>
  %x1 = arm.get_register : !arm.reg<x1>

  // CHECK: %x2 = arm.get_register : !arm.reg<x2>
  %x2 = arm.get_register : !arm.reg<x2>

  // CHECK: arm_func.beq %x1, %x2, ^loop {"comment" = "branch if equal"}
  // CHECK-ASM: beq %x1, %x2, ^loop # branch if equal

  arm_cf.beq %x1, %x2, ^loop {"comment" = "branch if equal"} : (!arm.reg<x1>, !arm.reg<x2>) -> ()

  // Define the loop block
  ^loop:
    arm.label "loop"

  // CHECK-GENERIC: "arm_cf.beq"(%x1, %x2, ^loop) {"comment" = "branch if equal"} : (!arm.register, !arm.register) -> ()
}
