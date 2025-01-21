"builtin.module"() ({
  "func.func"() <{function_type = (index) -> (), sym_name = "nested_loop_invariant"}> ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 100 : index}> : () -> index
    "scf.for"(%0, %2, %1) ({
    ^bb0(%arg1: index):
      %3 = "test.op"() : () -> i1
      %4 = "scf.if"(%3) ({
        %5 = "arith.muli"(%arg0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.yield"(%5) : (index) -> ()
      }, {
        "scf.yield"(%arg0) : (index) -> ()
      }) : (i1) -> index
      "test.op"(%4) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
