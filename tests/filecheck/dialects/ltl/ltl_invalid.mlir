// RUN: xdsl-opt %s --verify-diagnostics


builtin.module {
    %seq, %seq2 = "test.op"() : () -> (!ltl.sequence, !ltl.property)
    "ltl.and"(%seq, %seq2) : (!ltl.sequence,!ltl.property)->(!ltl.property)
    // CHECK: attribute !ltl.sequence expected from variable 'T', but got !ltl.property
}
