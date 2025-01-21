"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
  "cf.assert"(%0) <{msg = "some message"}> : (i1) -> ()
}) : () -> ()
