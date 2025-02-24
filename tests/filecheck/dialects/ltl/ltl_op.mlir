// RUN: XDSL_ROUNDTRIP

builtin.module {
    %b, %seq, %p = "test.op"() : () -> (i1, !ltl.sequence, !ltl.property)
    "ltl.and"(%seq, %seq) : (!ltl.sequence, !ltl.sequence) -> !ltl.sequence
    "ltl.and"(%p, %p) : (!ltl.property, !ltl.property) -> !ltl.property
    "ltl.and"(%b, %b) : (i1, i1) -> i1
}

// CHECK:       builtin.module {
// CHECK-NEXT:        %b, %seq, %p = "test.op"() : () -> (i1, !ltl.sequence, !ltl.property)
// CHECK-NEXT:        "ltl.and"(%seq, %seq) : (!ltl.sequence, !ltl.sequence) -> !ltl.sequence
// CHECK-NEXT:        "ltl.and"(%p, %p) : (!ltl.property, !ltl.property) -> !ltl.property
// CHECK-NEXT:        "ltl.and"(%b, %b) : (i1, i1) -> i1
// CHECK-NEXT:  }
