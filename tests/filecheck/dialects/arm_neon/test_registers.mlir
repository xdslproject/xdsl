// RUN: XDSL_ROUNDTRIP


// CHECK: "test.op"() {unallocated = !arm_neon.neon128reg} : () -> ()
"test.op"() {unallocated = !arm_neon.neon128reg} : () -> ()

// CHECK: "test.op"() {allocated = !arm_neon.neon128reg<v1>} : () -> ()
"test.op"() {allocated = !arm_neon.neon128reg<v1>} : () -> ()
