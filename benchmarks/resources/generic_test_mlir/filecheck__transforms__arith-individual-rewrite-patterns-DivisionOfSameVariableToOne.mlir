"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "hello"}> ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %1 = "arith.muli"(%arg0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %3 = "arith.divui"(%1, %2) : (i32, i32) -> i32
    "func.return"(%3) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
