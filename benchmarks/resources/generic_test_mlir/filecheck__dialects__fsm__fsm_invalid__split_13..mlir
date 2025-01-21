"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "test.op"() : () -> i1
    "fsm.state"() ({
      "fsm.output"() : () -> ()
    }, {
      "fsm.transition"() ({
      ^bb0(%arg2: i2):
        "fsm.return"() : () -> ()
      }, {
      ^bb0(%arg1: i1):
        "fsm.update"(%0, %arg1) : (i1, i1) -> ()
        "fsm.output"() : () -> ()
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
  }) {function_type = () -> (), initialState = "A", sym_name = "foo"} : () -> ()
}) : () -> ()
