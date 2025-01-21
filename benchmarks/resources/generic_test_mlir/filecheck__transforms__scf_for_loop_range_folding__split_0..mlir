"builtin.module"() ({
  %0:5 = "test.op"() : () -> (index, index, index, index, index)
  "scf.for"(%0#2, %0#3, %0#4) ({
  ^bb0(%arg3: index):
    %7 = "arith.addi"(%0#0, %arg3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %8 = "arith.muli"(%7, %0#1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "test.op"(%8) : (index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0#2, %0#3, %0#4) ({
  ^bb0(%arg2: index):
    %4 = "arith.addi"(%0#0, %arg2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %5 = "arith.addi"(%0#1, %0#1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %6 = "arith.muli"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "test.op"(%6) : (index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %1:2 = "test.op"() : () -> (index, index)
  "scf.for"(%0#2, %0#3, %0#4) ({
  ^bb0(%arg0: index):
    "scf.for"(%arg0, %1#0, %1#1) ({
    ^bb0(%arg1: index):
      %2 = "arith.addi"(%0#0, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.muli"(%2, %0#1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%3) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
