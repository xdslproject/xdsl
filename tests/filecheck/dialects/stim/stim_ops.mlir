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

// Single-qubit gate
stim.h [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.h [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.h"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

// Two-qubit gate
stim.cx [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.cx [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.cx"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

// Measurement
stim.m [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.m [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.m"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

// Measurement with flip probability
stim.mx(#builtin.float_data<0.01>) [!stim.qubit<0>]
// CHECK-NEXT:    stim.mx(#builtin.float_data<0.01>) [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.mx"() <{flip_probability = #builtin.float_data<0.01>, targets = [!stim.qubit<0>]}> : () -> ()

// Reset
stim.r [!stim.qubit<0>]
// CHECK-NEXT:    stim.r [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.r"() <{targets = [!stim.qubit<0>]}> : () -> ()

// Measure-reset
stim.mr [!stim.qubit<0>]
// CHECK-NEXT:    stim.mr [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.mr"() <{targets = [!stim.qubit<0>]}> : () -> ()

// Measure-reset with flip probability
stim.mry(#builtin.float_data<0.05>) [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.mry(#builtin.float_data<0.05>) [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.mry"() <{flip_probability = #builtin.float_data<0.05>, targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

// Tick annotation
stim.tick
// CHECK-NEXT:    stim.tick
// CHECK-GENERIC-NEXT:    "stim.tick"() : () -> ()

// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()
