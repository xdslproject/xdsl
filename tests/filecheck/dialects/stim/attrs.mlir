// RUN: XDSL_ROUNDTRIP

"test.op"() {
    qubitcoord = #stim.qubit_coord<0,0>
} : () -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {"qubitcoord" = #stim.qubit_coord<0, 0>} : () -> ()
// CHECK-NEXT:  }
