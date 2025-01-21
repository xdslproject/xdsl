"builtin.module"() ({
  %0 = "arith.constant"() <{value = false}> : () -> i1
  "scf.if"(%0) ({
    "scf.yield"() : () -> ()
  }, {
    "scf.yield"() : () -> ()
  }) : (i1) -> ()
}) : () -> ()
