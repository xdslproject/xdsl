"builtin.module"() ({
  %0:2 = "test.op"() : () -> (i1, i1)
  %1 = "mod_arith.add"(%0#0, %0#1) {modulus = 17 : i64} : (i1, i1) -> i1
}) : () -> ()
