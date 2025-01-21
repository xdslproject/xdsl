"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    "fsm.state"() ({
      "fsm.output"(%arg0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
  }) {function_type = (i16) -> i16, initialState = "A", sym_name = "foo"} : () -> ()
}) : () -> ()
