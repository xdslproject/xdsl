// RUN XDSL_ROUNDTRIP

"builtin.module"() ({
    %b = "arith.constant"() {value = 0} : () -> i1
    %seq = "arith.constant"() {value = 0} : () -> !ltl.sequence
    %p = "arith.constant"() {value = 0} : () -> !ltl.property
    "ltl.and"(%seq, %seq) : (!ltl.sequence,!ltl.sequence)->(!ltl.sequence)
    "ltl.and"(%p, %p) : (!ltl.property,!ltl.property)->(!ltl.property)
    "ltl.and"(%b, %b) : (i1,i1)->(i1)
}) : () -> ()

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:        %true = "arith.constant"() {value = true} : () -> i1
// CHECK-NEXT:        %seq = "arith.constant"() {value = 0} : () -> !ltl.sequence
// CHECK-NEXT:        %p = "arith.constant"() {value = 0} : () -> !ltl.property
// CHECK-NEXT:        "ltl.and"(%true, %true) : (i1,i1)->(i1)
// CHECK-NEXT:        "ltl.and"(%seq, %seq) : (!ltl.sequence,!ltl.sequence)->(!ltl.sequence)
// CHECK-NEXT:        "ltl.and"(%p, %p) : (!ltl.property,!ltl.property)->(!ltl.property)
// CHECK-NEXT:  }) : () -> ()
