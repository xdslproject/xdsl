"builtin.module"() ({
  "test.op"() ({
  ^bb0(%arg5: i32):
    "test.termop"(%arg5) : (i32) -> ()
  }) : () -> ()
  "test.op"() ({
  ^bb0(%arg4: i64):
    "test.termop"(%arg4) : (i64) -> ()
  }) : () -> ()
  "test.op"() ({
  ^bb0(%arg1: i1):
    "test.op"(%arg1) ({
    ^bb0(%arg3: i32):
      "test.termop"() : () -> ()
    }, {
    ^bb0(%arg2: i32):
      "test.termop"() : () -> ()
    }) : (i1) -> ()
    "test.termop"(%arg1) : (i1) -> ()
  }) : () -> ()
  "test.op"() ({
  ^bb0(%arg0: i1):
    "test.op"(%arg0) ({
      %1 = "test.termop"() : () -> i32
    }, {
      "test.termop"() : () -> ()
    }) : (i1) -> ()
    %0 = "test.op"() : () -> i32
    "test.termop"(%arg0) : (i1) -> ()
  }) : () -> ()
}) : () -> ()
