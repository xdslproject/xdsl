"builtin.module"() ({
  %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
  "fsm.machine"() ({
    "fsm.state"() ({
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
  }) {function_type = (i16) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
  %1 = "arith.constant"() <{value = 0 : i16}> : () -> i16
  %2 = "arith.constant"() <{value = 0 : i16}> : () -> i16
  %3 = "arith.constant"() <{value = 0 : i16}> : () -> i16
  %4 = "fsm.hw_instance"(%3, %1, %2) {machine = @foo, sym_name = "foo_inst"} : (i16, i16, i16) -> i16
}) : () -> ()
