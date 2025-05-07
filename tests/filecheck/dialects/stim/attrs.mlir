// RUN: XDSL_ROUNDTRIP

"test.op"() {
    qubit = !stim.qubit<0>,
    qubitcoord = #stim.qubit_coord<(0,0), !stim.qubit<0>>
} : () -> ()

%qubit0 = "test.op"() : () -> (!stim.qubit<0>)

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {qubit = !stim.qubit<0>, qubitcoord = #stim.qubit_coord<(0, 0), !stim.qubit<0>>} : () -> ()
// CHECK-NEXT:    %qubit0 = "test.op"() : () -> !stim.qubit<0>
// CHECK-NEXT:  }
