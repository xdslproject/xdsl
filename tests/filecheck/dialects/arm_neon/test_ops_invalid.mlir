// RUN: xdsl-opt -t arm-asm %s --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
    %x1, %v1, %v2, %v3, %v4, %v5 = "test.op"() : () -> (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>)
    
    // CHECK:       Operation does not verify: src_regs must contain between 1 and 4 elements, but got 5.
    arm_neon.dvars.st1 %v1, %v2, %v3, %v4, %v5 [%x1] S {comment = "st1 op"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>) -> !arm.reg<x1>
}

// -----

builtin.module {
    %x1, %v1, %v2, %v3, %v4, %v5 = "test.op"() : () -> (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>)

    // CHECK:       Operation does not verify: dest_regs must contain between 1 and 4 elements, but got 5.
    arm_neon.dvars.ld1 %v1, %v2, %v3, %v4, %v5 [%x1] S {comment = "ld1 op"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>) -> !arm.reg<x1>
}
