"builtin.module"() ({
  %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
  "fsm.machine"() ({
    "fsm.state"() ({
      "fsm.output"(%0) : (i16) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
    "fsm.state"() ({
      "fsm.output"(%0) : (i16) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "B"} : () -> ()
    "fsm.state"() ({
      "fsm.output"(%0) : (i16) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "C"} : () -> ()
  }) {function_type = (i16) -> i16, initialState = "A", sym_name = "foo"} : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "qux"}> ({
    %1 = "arith.constant"() <{value = 16 : i16}> : () -> i16
    %2 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
    %3 = "arith.constant"() <{value = true}> : () -> i1
    %4 = "fsm.trigger"(%3, %2) : (i1, !fsm.instancetype) -> i16
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
