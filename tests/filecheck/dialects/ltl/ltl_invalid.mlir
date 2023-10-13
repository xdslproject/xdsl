// RUN XDSL_ROUNDTRIP


"builtin.module"() ({
    %true = "arith.constant"() {value = true} : () -> i1
    %seq = "arith.constant"() {value = 0} : () -> !ltl.sequence
    %seq2 = "arith.constant"() {value = 0} : () -> !ltl.property
    "ltl.and"(%seq, %seq2) : (!ltl.sequence,!ltl.property)->(!ltl.property)
    // CHECK: attribute !ltl.sequence expected from variable 'T', but got !ltl.property
}) : () -> ()
