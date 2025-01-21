"builtin.module"() ({
  %0 = "arith.constant"() <{value = 2 : i16}> : () -> i16
  "fsm.machine"() ({
    "fsm.state"() ({
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
    "fsm.state"() ({
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "B"} : () -> ()
    "fsm.state"() ({
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "C"} : () -> ()
  }) {function_type = (i16) -> i16, initialState = "A", sym_name = "foo"} : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "qux"}> ({
    %1 = "arith.constant"() <{value = 16 : i16}> : () -> i16
    %2 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
    %3 = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %4 = "fsm.trigger"(%3, %2) : (i16, !fsm.instancetype) -> i1
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
