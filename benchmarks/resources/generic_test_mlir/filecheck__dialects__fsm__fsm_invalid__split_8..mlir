"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
      "fsm.output"(%arg0) : (i1) -> ()
    }) {sym_name = "A"} : () -> ()
  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
}) : () -> ()
