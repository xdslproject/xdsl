"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "while"}> ({
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "scf.while"(%0) ({
    ^bb0(%arg1: i32):
      %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %3 = "arith.cmpi"(%2, %arg1) <{predicate = 1 : i64}> : (i32, i32) -> i1
      "scf.condition"(%3, %2) : (i1, i32) -> ()
    }, {
    ^bb0(%arg0: i32):
      "scf.yield"(%arg0) : (i32) -> ()
    }) : (i32) -> i32
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
