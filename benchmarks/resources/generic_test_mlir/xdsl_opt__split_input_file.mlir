"builtin.module"() ({
  "builtin.module"() ({
  ^bb0:
  }) : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    %1 = "test.op"() : () -> i1
  }) : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i2
  }) : () -> ()
}) : () -> ()
