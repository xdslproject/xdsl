// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM
// CHECK: %v1 = arm_neon.get_register : !arm_neon.reg<v1>
%v1 = arm_neon.get_register : !arm_neon.reg<v1>
// CHECK: %v2 = arm_neon.get_register : !arm_neon.reg<v2>
%v2 = arm_neon.get_register : !arm_neon.reg<v2>
// CHECK: %dss_fmulvec = arm_neon.dss.fmulvec %v1, %v2 {arrangement = "4S", comment = "floating-point vector multiply v1 by v2", scalar_idx = 0 : i8} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>
// CHECK-ASM: fmul v3.4S, v1.4S, v2.S[0] # floating-point vector multiply v1 by v2
%dss_fmulvec = arm_neon.dss.fmulvec %v1, %v2 {"arrangement" = "4S", "comment" = "floating-point vector multiply v1 by v2", "scalar_idx" = 0 : i8} : (!arm_neon.reg<v1>, !arm_neon.reg<v2>) -> !arm_neon.reg<v3>
