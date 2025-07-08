// RUN: XDSL_ROUNDTRIP

"test.op"() {
    signed = dense<127> : tensor<si8>,
    signless = dense<255> : tensor<i8>,
    unsigned = dense<255> : tensor<ui8>
} : () -> ()

//      CHECK: signed = dense<127> : tensor<si8>
// CHECK-SAME: signless = dense<-1> : tensor<i8>
// CHECK-SAME: unsigned = dense<255> : tensor<ui8>
