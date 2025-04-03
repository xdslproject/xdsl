// RUN: xdsl-opt -t arm-asm %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
    %x1 = "arm.get_register"() : () -> !arm.reg<x1>
    %v1 = "arm_neon.get_register"() : () -> !arm_neon.reg<v1>
    %v2 = "arm_neon.get_register"() : () -> !arm_neon.reg<v2>
    %v3 = "arm_neon.get_register"() : () -> !arm_neon.reg<v3>
    %v4 = "arm_neon.get_register"() : () -> !arm_neon.reg<v4>
    %v5 = "arm_neon.get_register"() : () -> !arm_neon.reg<v5>
    "arm_neon.dvars.st1"(%x1, %v1, %v2, %v3, %v4, %v5) {arrangement = #arm_neon<arrangement S>, comment = "st1 op"} : (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>) -> ()
}) : () -> ()

// CHECK:       Operation does not verify: src_regs must contain between 1 and 4 elements, but got 5.

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    %x1 = "arm.get_register"() : () -> !arm.reg<x1>
// CHECK-NEXT:    %v1 = "arm_neon.get_register"() : () -> !arm_neon.reg<v1>
// CHECK-NEXT:    %v2 = "arm_neon.get_register"() : () -> !arm_neon.reg<v2>
// CHECK-NEXT:    %v3 = "arm_neon.get_register"() : () -> !arm_neon.reg<v3>
// CHECK-NEXT:    %v4 = "arm_neon.get_register"() : () -> !arm_neon.reg<v4>
// CHECK-NEXT:    %v5 = "arm_neon.get_register"() : () -> !arm_neon.reg<v5>
// CHECK-NEXT:    "arm_neon.dvars.st1"(%x1, %v1, %v2, %v3, %v4, %v5) {arrangement = #arm_neon<arrangement S>, comment = "st1 op"} : (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>) -> ()
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^-------------------------------------------------------------------
// CHECK-NEXT:    | Operation does not verify: src_regs must contain between 1 and 4 elements, but got 5.
// CHECK-NEXT:    ---------------------------------------------------------------------------------------
// CHECK-NEXT:  }) : () -> ()
