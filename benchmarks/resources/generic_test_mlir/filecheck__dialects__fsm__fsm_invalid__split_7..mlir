"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      "fsm.output"() : () -> ()
    }, {
    }) {sym_name = "B"} : () -> ()
  }) {function_type = () -> (), initialState = "B", res_attrs = [{name = "1", type = "2"}, {name = "3", type = "4"}], res_names = ["names"], sym_name = "foo"} : () -> ()
}) : () -> ()
