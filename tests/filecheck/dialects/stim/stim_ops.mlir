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

stim.h [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.h [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.h"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.s [!stim.qubit<0>]
// CHECK-NEXT:    stim.s [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.s"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.s_dag [!stim.qubit<0>]
// CHECK-NEXT:    stim.s_dag [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.s_dag"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.x [!stim.qubit<0>]
// CHECK-NEXT:    stim.x [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.x"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.y [!stim.qubit<0>]
// CHECK-NEXT:    stim.y [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.y"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.z [!stim.qubit<0>]
// CHECK-NEXT:    stim.z [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.z"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.i [!stim.qubit<0>]
// CHECK-NEXT:    stim.i [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.i"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.sqrt_x [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK-NEXT:    stim.sqrt_x [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK-GENERIC-NEXT:    "stim.sqrt_x"() <{targets = [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]}> : () -> ()

stim.sqrt_x_dag [!stim.qubit<0>]
// CHECK-NEXT:    stim.sqrt_x_dag [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.sqrt_x_dag"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.sqrt_y [!stim.qubit<0>]
// CHECK-NEXT:    stim.sqrt_y [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.sqrt_y"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.sqrt_y_dag [!stim.qubit<0>]
// CHECK-NEXT:    stim.sqrt_y_dag [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.sqrt_y_dag"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.cx [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.cx [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.cx"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.cy [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.cy [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.cy"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.cz [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.cz [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.cz"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.swap [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.swap [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.swap"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.iswap [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.iswap [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.iswap"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.iswap_dag [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.iswap_dag [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.iswap_dag"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.m [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.m [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.m"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.m flip_prob #builtin.float_data<0.01> [!stim.qubit<0>]
// CHECK-NEXT:    stim.m flip_prob #builtin.float_data<0.01> [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.m"() <{flip_prob = #builtin.float_data<0.01>, targets = [!stim.qubit<0>]}> : () -> ()

stim.mx [!stim.qubit<0>]
// CHECK-NEXT:    stim.mx [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.mx"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.my flip_prob #builtin.float_data<0.05> [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.my flip_prob #builtin.float_data<0.05> [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.my"() <{flip_prob = #builtin.float_data<0.05>, targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.r [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.r [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.r"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.rx [!stim.qubit<0>]
// CHECK-NEXT:    stim.rx [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.rx"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.ry [!stim.qubit<0>]
// CHECK-NEXT:    stim.ry [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.ry"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.mr [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-NEXT:    stim.mr [!stim.qubit<0>, !stim.qubit<1>]
// CHECK-GENERIC-NEXT:    "stim.mr"() <{targets = [!stim.qubit<0>, !stim.qubit<1>]}> : () -> ()

stim.mr flip_prob #builtin.float_data<0.01> [!stim.qubit<0>]
// CHECK-NEXT:    stim.mr flip_prob #builtin.float_data<0.01> [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.mr"() <{flip_prob = #builtin.float_data<0.01>, targets = [!stim.qubit<0>]}> : () -> ()

stim.mrx [!stim.qubit<0>]
// CHECK-NEXT:    stim.mrx [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.mrx"() <{targets = [!stim.qubit<0>]}> : () -> ()

stim.mry flip_prob #builtin.float_data<0.1> [!stim.qubit<0>]
// CHECK-NEXT:    stim.mry flip_prob #builtin.float_data<0.1> [!stim.qubit<0>]
// CHECK-GENERIC-NEXT:    "stim.mry"() <{flip_prob = #builtin.float_data<0.1>, targets = [!stim.qubit<0>]}> : () -> ()
// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()
