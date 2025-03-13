// RUN: XDSL_ROUNDTRIP


// CHECK: "test.op"() {unallocated = !arm_neon.reg} : () -> ()
"test.op"() {unallocated = !arm_neon.reg} : () -> ()

// CHECK: "test.op"() {allocated = !arm_neon.reg<v1>} : () -> ()
"test.op"() {allocated = !arm_neon.reg<v1>} : () -> ()
