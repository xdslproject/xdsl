"builtin.module"() ({
  "func.func"() <{function_type = (index) -> index, sym_name = "test"}> ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %1 = "arith.muli"(%arg0, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "func.return"(%1) : (index) -> ()
  }) : () -> ()
}) : () -> ()
