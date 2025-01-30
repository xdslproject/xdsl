// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({

stim.circuit {}
// CHECK-NEXT:    stim.circuit {
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-GENERIC-NEXT:   }) : () -> ()

stim.circuit attributes {"hello" = "world"} {}
// CHECK-NEXT:    stim.circuit attributes {hello = "world"} {
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-GENERIC-NEXT:   }) {hello = "world"} : () -> ()

stim.assign_qubit_coord <(0, 0), !stim.qubit<0>>
// CHECK-NEXT:    stim.assign_qubit_coord <(0, 0), !stim.qubit<0>>
// CHECK-GENERIC-NEXT:    "stim.assign_qubit_coord"() <{qubitmapping = #stim.qubit_coord<(0, 0), !stim.qubit<0>>}> : () -> ()

stim.circuit {stim.assign_qubit_coord <(0, 0), !stim.qubit<0>>}
// CHECK-NEXT:    stim.circuit {
// CHECK-NEXT:  stim.assign_qubit_coord <(0, 0), !stim.qubit<0>>
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-GENERIC-NEXT:  "stim.assign_qubit_coord"() <{qubitmapping = #stim.qubit_coord<(0, 0), !stim.qubit<0>>}>
// CHECK-GENERIC-NEXT:  }) : () -> ()

stim.circuit attributes {"hello" = "world"} {stim.assign_qubit_coord <(0, 0), !stim.qubit<0>>}
// CHECK-NEXT:    stim.circuit attributes {hello = "world"} {
// CHECK-NEXT:  stim.assign_qubit_coord <(0, 0), !stim.qubit<0>>
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-GENERIC-NEXT:  "stim.assign_qubit_coord"() <{qubitmapping = #stim.qubit_coord<(0, 0), !stim.qubit<0>>}>
// CHECK-GENERIC-NEXT:  }) {hello = "world"} : () -> ()

stim.circuit qubitlayout [#stim.qubit_coord<(0, 0), !stim.qubit<0>>] attributes {"hello" = "world"} {stim.assign_qubit_coord <(1, 2), !stim.qubit<2>>}
// CHECK-NEXT:    stim.circuit qubitlayout [#stim.qubit_coord<(0, 0), !stim.qubit<0>>] attributes {hello = "world"}
// CHECK-NEXT:  stim.assign_qubit_coord <(1, 2), !stim.qubit<2>>
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() <{qubitlayout = [#stim.qubit_coord<(0, 0), !stim.qubit<0>>]}> ({
// CHECK-GENERIC-NEXT:  "stim.assign_qubit_coord"() <{qubitmapping = #stim.qubit_coord<(1, 2), !stim.qubit<2>>}>
// CHECK-GENERIC-NEXT:  }) {hello = "world"} : () -> ()

// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()
