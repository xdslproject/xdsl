"builtin.module"() ({
  "test.op"() ({
    %4 = "test.pureop"() : () -> i32
    %5 = "test.pureop"(%4) : (i32) -> i32
    "test.termop"() : () -> ()
  }) : () -> ()
  "test.op"() ({
    "test.termop"()[^bb2] : () -> ()
  ^bb1:  // no predecessors
    "test.op"() : () -> ()
    "test.termop"()[^bb2] : () -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "test.termop"() : () -> ()
  }) : () -> ()
  "test.op"() ({
    %2 = "test.pureop"(%3) : (i32) -> i32
    %3 = "test.pureop"(%2) : (i32) -> i32
    "test.termop"() : () -> ()
  }) : () -> ()
  "test.op"() ({
    "test.termop"() : () -> ()
  ^bb1:  // pred: ^bb2
    %0 = "test.op"(%1) : (i32) -> i32
    "test.termop"()[^bb2] : () -> ()
  ^bb2:  // pred: ^bb1
    %1 = "test.op"(%0) : (i32) -> i32
    "test.termop"()[^bb1] : () -> ()
  }) : () -> ()
  "test.op"() ({
    "test.op"() ({
      "test.pureop"() : () -> ()
      "test.termop"() : () -> ()
    }) : () -> ()
    "test.termop"() : () -> ()
  }) : () -> ()
}) : () -> ()
