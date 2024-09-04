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
// CHECK-NEXT:    stim.circuit attributes {"hello" = "world"} {
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-GENERIC-NEXT:   }) {"hello" = "world"} : () -> ()

%q0 = qref.alloc<1>
stim.assign_qubit_coord <0, 0> %q0
// CHECK-NEXT:  %q0 = qref.alloc<1>
// CHECK-NEXT:    stim.assign_qubit_coord <0, 0> %q0
// CHECK-GENERIC-NEXT: %q0 = "qref.alloc"() : () -> !qref.qubit
// CHECK-GENERIC-NEXT:    "stim.assign_qubit_coord"(%q0) <{"qubitcoord" = #stim.qubit_coord<0, 0>}> : (!qref.qubit) -> ()

stim.circuit {
    %q1 = qref.alloc<1>
    stim.assign_qubit_coord <0, 0> %q1
    stim.clifford I X dag (%q1)
    stim.tick
    %r0 = stim.measure X (%q1)
    stim.reset X (%q1)
}
// CHECK-NEXT:    stim.circuit {
// CHECK-NEXT:  %q1 = qref.alloc<1>
// CHECK-NEXT:    stim.assign_qubit_coord <0, 0> %q1
// CHECK-NEXT:  stim.clifford I X dag (%q1)
// CHECK-NEXT:  stim.tick
// CHECK-NEXT:  %r0 = stim.measure X (%q1)
// CHECK-NEXT:  stim.reset X (%q1)
// CHECK-NEXT: }
// CHECK-GENERIC-NEXT:    "stim.circuit"() ({
// CHECK-GENERIC-NEXT: %q1 = "qref.alloc"() : () -> !qref.qubit
// CHECK-GENERIC-NEXT:    "stim.assign_qubit_coord"(%q1) <{"qubitcoord" = #stim.qubit_coord<0, 0>}> : (!qref.qubit) -> ()
// CHECK-GENERIC-NEXT:  "stim.clifford"(%q1) <{"gate_name" = #stim.singlequbitclifford I, "pauli_modifier" = #stim.pauli X, "dag"}> : (!qref.qubit) -> ()
// CHECK-GENERIC-NEXT: "stim.tick"() : () -> ()
// CHECK-GENERIC-NEXT: %r0 = "stim.measure"(%q1) <{"pauli_modifier" = #stim.pauli X}> : (!qref.qubit) -> i1
// CHECK-GENERIC-NEXT: "stim.reset"(%q1) <{"pauli_modifier" = #stim.pauli X}> : (!qref.qubit) -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()
