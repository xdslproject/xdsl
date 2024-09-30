// RUN: XDSL_ROUNDTRIP

%lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)

// CHECK: %add_res = mod_arith.add %lhsi1, %rhsi1 {"modulus" = 17 : i64} : i1
%add_res = "mod_arith.add"(%lhsi1, %rhsi1) {modulus=17} : (i1, i1) -> i1
