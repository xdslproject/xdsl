"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "bb_ordered_by_dominance"}> ({
    "test.termop"()[^bb1] : () -> ()
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %2 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    "test.termop"()[^bb2] : () -> ()
  ^bb2:  // pred: ^bb1
    %3 = "arith.addi"(%2, %2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    "test.termop"()[^bb1] : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "bb_ordered_against_dominance"}> ({
    "test.termop"()[^bb2] : () -> ()
  ^bb1:  // pred: ^bb2
    %0 = "arith.addi"(%1, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    "test.termop"()[^bb2] : () -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %1 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    "test.termop"()[^bb1] : () -> ()
  }) : () -> ()
}) : () -> ()
