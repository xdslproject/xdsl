"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "v1"} : () -> i16
    %1 = "fsm.variable"() {initValue = 2 : i16, name = "v2"} : () -> i16
    "fsm.state"() ({
      "fsm.output"() : () -> ()
    }, {
      "fsm.transition"() ({
      }, {
      ^bb0(%arg1: i1):
        "fsm.update"(%0, %1) {value = "v2", variable = "v1"} : (i16, i16) -> ()
        "fsm.update"(%0, %1) {value = "v2", variable = "v1"} : (i16, i16) -> ()
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
  }) {function_type = () -> (), initialState = "A", res_attrs = [{name = "1", type = "2"}], res_names = ["names"], sym_name = "foo"} : () -> ()
}) : () -> ()
