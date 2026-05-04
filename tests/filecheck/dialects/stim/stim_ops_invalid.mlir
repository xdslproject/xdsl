// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

stim.cx [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK: Expected an even number of targets for CX, got 3

// -----

stim.cy [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK: Expected an even number of targets for CY, got 3

// -----

stim.cz [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK: Expected an even number of targets for CZ, got 3

// -----

stim.swap [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK: Expected an even number of targets for SWAP, got 3

// -----

stim.iswap [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK: Expected an even number of targets for ISWAP, got 3

// -----

stim.iswap_dag [!stim.qubit<0>, !stim.qubit<1>, !stim.qubit<2>]
// CHECK: Expected an even number of targets for ISWAP_DAG, got 3
