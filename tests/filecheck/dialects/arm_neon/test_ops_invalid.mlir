// RUN: xdsl-opt -t arm-asm %s --verify-diagnostics --parsing-diagnostics --split-input-file | filecheck %s

builtin.module {
    %x1, %v1, %v2, %v3, %v4, %v5 = "test.op"() : () -> (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>)
    
    // CHECK:       Operation does not verify: src_regs must contain between 1 and 4 elements, but got 5.
    arm_neon.dvars.st1 %v1, %v2, %v3, %v4, %v5 [%x1] S {comment = "st1 op"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>, !arm_neon.reg<v5>) -> !arm.reg<x1>
}

// -----

builtin.module {
    %x1 = "test.op"() : () -> (!arm.reg<x1>)

    // CHECK:       Operation does not verify: dest_regs must contain between 1 and 4 elements, but got 5.
    %v11, %v12, %v13, %v14, %v15 = arm_neon.dvars.ld1 [%x1] S {comment = "ld1 op"} : !arm.reg<x1> -> (!arm_neon.reg<v11>, !arm_neon.reg<v12>, !arm_neon.reg<v13>, !arm_neon.reg<v14>, !arm_neon.reg<v15>)
}

// -----

builtin.module {
    %x1, %v1, %v2, %v3 = "test.op"() : () -> (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>)

    // CHECK: attribute !arm_neon.reg<v3> expected from variable 'SAME_NEON_REGISTER_TYPE', but got !arm_neon.reg<v4>
    %dss_fmla = "arm_neon.dss.fmla"(%v3, %v1, %v2) {scalar_idx = 0 : i8, arrangement = #arm_neon<arrangement S>, "comment" = "Floating-point fused Multiply-Add to accumulator"} : (!arm_neon.reg<v3>, !arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v4>
}


// -----

builtin.module {
    %x1, %v1, %v2, %v3 = "test.op"() : () -> (!arm.reg<x1>, !arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>)

    // CHECK: operand is used with type !arm_neon.reg<v4>, but has been previously used or defined with type !arm_neon.reg<v3>
    %dss_fmla = arm_neon.dss.fmla %v3, %v1, %v2[0] S {"comment" = "Floating-point fused Multiply-Add to accumulator"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v4>
}
