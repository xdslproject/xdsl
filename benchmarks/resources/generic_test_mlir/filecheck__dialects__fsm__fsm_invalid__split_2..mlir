"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      "fsm.output"() : () -> ()
    }, {
      "fsm.transition"() ({
      ^bb0(%arg1: i1):
        "test.termop"() : () -> ()
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
  }) {function_type = () -> (), initialState = "A", sym_name = "foo"} : () -> ()
}) : () -> ()
