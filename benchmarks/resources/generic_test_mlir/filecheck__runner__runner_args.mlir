"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "one"}> ({
    %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "func.return"(%3) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "two"}> ({
    %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "void"}> ({
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (i32, i32), sym_name = "tuple"}> ({
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) : () -> ()
}) : () -> ()
