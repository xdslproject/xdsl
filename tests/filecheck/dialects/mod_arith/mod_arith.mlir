// RUN: XDSL_ROUNDTRIP

%lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)

// CHECK: %add_res = "mod_arith.add"(%lhsi1, %rhsi1) : (i1, i1) -> i1
%add_res = "mod_arith.add"(%lhsi1, %rhsi1) : (i1, i1) -> i1
