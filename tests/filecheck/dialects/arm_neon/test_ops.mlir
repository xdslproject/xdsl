// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM

// CHECK: %x1 = arm.get_register : !arm.reg<x1>
%x1 = arm.get_register : !arm.reg<x1>
// CHECK: %v1 = arm_neon.get_register : !arm_neon.reg<v1>
%v1 = arm_neon.get_register : !arm_neon.reg<v1>
// CHECK: %v2 = arm_neon.get_register : !arm_neon.reg<v2>
%v2 = arm_neon.get_register : !arm_neon.reg<v2>
// CHECK: %v3 = arm_neon.get_register : !arm_neon.reg<v3>
%v3 = arm_neon.get_register : !arm_neon.reg<v3>
// CHECK: %v4 = arm_neon.get_register : !arm_neon.reg<v4>
%v4 = arm_neon.get_register : !arm_neon.reg<v4>

// CHECK: %dss_fmul_mixed = arm_neon.dss.fmul %v1, %v2[0] S {comment = "floating-point vector multiply (mixed) v1 by v2"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>
// CHECK-ASM: fmul v3.4S, v1.4S, v2.S[0] # floating-point vector multiply (mixed) v1 by v2
%dss_fmul_mixed = arm_neon.dss.fmul %v1, %v2[0] S {"comment" = "floating-point vector multiply (mixed) v1 by v2"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>

// CHECK: %dss_fmul_vec = arm_neon.dss.fmul %v1, %v2 S {comment = "floating-point vector multiply v1 by v2"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>
// CHECK-ASM: fmul v3.4S, v1.4S, v2.4S # floating-point vector multiply v1 by v2
%dss_fmul_vec = arm_neon.dss.fmul %v1, %v2 S {"comment" = "floating-point vector multiply v1 by v2"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>

// CHECK: %dss_fmla = arm_neon.dss.fmla %v3, %v1, %v2[0] S {comment = "Floating-point fused Multiply-Add to accumulator"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>
// CHECK-ASM: fmla v3.4S, v1.4S, v2.S[0] # Floating-point fused Multiply-Add to accumulator
%dss_fmla = arm_neon.dss.fmla %v3, %v1, %v2[0] S {"comment" = "Floating-point fused Multiply-Add to accumulator"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>

// CHECK: arm_neon.dvars.st1 %v1, %v2, %v3, %v4 [%x1] S {comment = "st1 op"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>) -> !arm.reg<x1>
// CHECK-ASM: st1 {v1.4S, v2.4S, v3.4S, v4.4S}, [x1] # st1 op
arm_neon.dvars.st1 %v1, %v2, %v3, %v4 [%x1] S {comment = "st1 op"} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>, !arm_neon.reg<v3>, !arm_neon.reg<v4>) -> !arm.reg<x1>

// CHECK: %v11, %v12, %v13, %v14 = arm_neon.dvars.ld1 [%x1] S {comment = "ld1 op"} : !arm.reg<x1> -> (!arm_neon.reg<v11>, !arm_neon.reg<v12>, !arm_neon.reg<v13>, !arm_neon.reg<v14>)
// CHECK-ASM: ld1 {v11.4S, v12.4S, v13.4S, v14.4S}, [x1] # ld1 op
%v11, %v12, %v13, %v14 = arm_neon.dvars.ld1 [%x1] S {comment = "ld1 op"} : !arm.reg<x1> -> (!arm_neon.reg<v11>, !arm_neon.reg<v12>, !arm_neon.reg<v13>, !arm_neon.reg<v14>)

// CHECK: %ds_dup = arm_neon.ds.dup %x1 S {comment = "Duplicate general-purpose register to vector"} : !arm.reg<x1> -> (!arm_neon.reg<v1>)
// CHECK-ASM: dup v1.4S, x1 # Duplicate general-purpose register to vector
%ds_dup = arm_neon.ds.dup %x1 S {comment = "Duplicate general-purpose register to vector"} : !arm.reg<x1> -> !arm_neon.reg<v1>

// CHECK: %dsvec_mov = arm_neon.dsvec.mov %v3[1] S {comment = "Set X1 to the value of the second single word in V3."} : !arm_neon.reg<v3> -> !arm.reg<x1>
// CHECK-ASM: mov x1, v3.S[1] # Set X1 to the value of the second single word in V3.
%dsvec_mov = arm_neon.dsvec.mov %v3[1] S {comment = "Set X1 to the value of the second single word in V3."} : !arm_neon.reg<v3> -> !arm.reg<x1>
