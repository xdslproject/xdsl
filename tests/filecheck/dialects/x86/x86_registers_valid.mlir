// RUN: XDSL_ROUNDTRIP


// CHECK: "test.op"() {unallocated = !x86.avx512maskreg} : () -> ()
"test.op"() {unallocated = !x86.avx512maskreg} : () -> ()

// CHECK: "test.op"() {allocated = !x86.avx512maskreg<k1>} : () -> ()
"test.op"() {allocated = !x86.avx512maskreg<k1>} : () -> ()
