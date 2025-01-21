"builtin.module"() ({
  %0:3 = "test.op"() : () -> (index, index, index)
  "scf.for"(%0#0, %0#1, %0#2) ({
  ^bb0(%arg0: index):
    "scf.for"(%0#0, %0#1, %0#2) ({
    ^bb0(%arg1: index):
      "scf.for"(%0#0, %0#1, %0#2) ({
      ^bb0(%arg2: index):
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
