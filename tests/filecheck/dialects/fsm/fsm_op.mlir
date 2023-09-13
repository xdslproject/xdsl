// RUN: XDSL_ROUNDTRIP

builtin.module {
    
    // CHECK: fsm.machine @foo(%arg0: i1) attributes {initialState = "IDLE"} {
    // CHECK:   fsm.state @IDLE
    // CHECK: }

    fsm.machine @foo(%arg0: i1) attributes {initialState = "IDLE"} {
    fsm.state @IDLE
    }

    // -----

}