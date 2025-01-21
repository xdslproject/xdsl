"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "hello"}> ({
  ^bb0(%arg0: i32):
    %0 = "arith.addi"(%arg0, %arg0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %1 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %2 = "arith.divui"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
