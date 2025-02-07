// RUN: XDSL_ROUNDTRIP


// CHECK: "test.op"() {unallocated = !arm.reg} : () -> ()
"test.op"() {unallocated = !arm.reg} : () -> ()

// CHECK: "test.op"() {allocated = !arm.reg<x1>} : () -> ()
"test.op"() {allocated = !arm.reg<x1>} : () -> ()
