// RUN: XDSL_ROUNDTRIP


// CHECK: "test.op"() {unallocated = !arm_neon.neonreg} : () -> ()
"test.op"() {unallocated = !arm_neon.neonreg} : () -> ()

// CHECK: "test.op"() {allocated = !arm_neon.neonreg<v1>} : () -> ()
"test.op"() {allocated = !arm_neon.neonreg<v1>} : () -> ()
