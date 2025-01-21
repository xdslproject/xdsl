"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
    ^bb0(%arg1: i1):
      "fsm.transition"() ({
      ^bb0(%arg2: i2):
        "fsm.return"() : () -> ()
      }, {
      }) {nextState = @A} : () -> ()
    }, {
    }) {sym_name = "A"} : () -> ()
  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
}) : () -> ()
