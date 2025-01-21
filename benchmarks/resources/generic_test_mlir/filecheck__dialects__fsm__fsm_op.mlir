"builtin.module"() ({
  "builtin.module"() ({
    "fsm.machine"() ({
    ^bb0(%arg1: i1):
      "fsm.state"() ({
      }, {
      }) {sym_name = "IDLE"} : () -> ()
    }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    %15 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.machine"() ({
      "fsm.state"() ({
        "fsm.output"(%15) : (i16) -> ()
      }, {
        "fsm.transition"() ({
        }, {
        }) {nextState = @A} : () -> ()
      }) {sym_name = "A"} : () -> ()
      "fsm.state"() ({
        "fsm.output"(%15) : (i16) -> ()
      }, {
        "fsm.transition"() ({
        }, {
        }) {nextState = @C} : () -> ()
      }) {sym_name = "B"} : () -> ()
      "fsm.state"() ({
        "fsm.output"(%15) : (i16) -> ()
      }, {
        "fsm.transition"() ({
        }, {
        }) {nextState = @C} : () -> ()
      }) {sym_name = "C"} : () -> ()
    }) {function_type = (i16) -> i16, initialState = "A", sym_name = "foo"} : () -> ()
    %16 = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %17 = "arith.constant"() <{value = 50 : i16}> : () -> i16
    %18 = "fsm.hw_instance"(%15, %16, %17) {machine = @foo, sym_name = "foo_inst"} : (i16, i16, i16) -> i16
  }) : () -> ()
  "builtin.module"() ({
    "fsm.machine"() ({
    ^bb0(%arg0: i1):
      %5 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
      "fsm.state"() ({
        %14 = "arith.constant"() <{value = true}> : () -> i1
        "fsm.output"(%arg0) : (i1) -> ()
      }, {
        "fsm.transition"() ({
          "fsm.return"(%arg0) : (i1) -> ()
        }, {
          %13 = "arith.constant"() <{value = 256 : i16}> : () -> i16
          "fsm.update"(%5, %13) : (i16, i16) -> ()
        }) {nextState = @BUSY} : () -> ()
      }) {sym_name = "IDLE"} : () -> ()
      "fsm.state"() ({
        %12 = "arith.constant"() <{value = false}> : () -> i1
        "fsm.output"(%arg0) : (i1) -> ()
      }, {
        "fsm.transition"() ({
          %10 = "arith.constant"() <{value = 0 : i16}> : () -> i16
          %11 = "arith.cmpi"(%5, %10) <{predicate = 1 : i64}> : (i16, i16) -> i1
          "fsm.return"(%11) : (i1) -> ()
        }, {
          %8 = "arith.constant"() <{value = 1 : i16}> : () -> i16
          %9 = "arith.subi"(%5, %8) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
          "fsm.update"(%5, %9) : (i16, i16) -> ()
        }) {nextState = @BUSY} : () -> ()
        "fsm.transition"() ({
          %6 = "arith.constant"() <{value = 0 : i16}> : () -> i16
          %7 = "arith.cmpi"(%5, %6) <{predicate = 0 : i64}> : (i16, i16) -> i1
          "fsm.return"(%7) : (i1) -> ()
        }, {
        }) {nextState = @IDLE} : () -> ()
      }) {sym_name = "BUSY"} : () -> ()
    }) {function_type = (i1) -> i1, initialState = "IDLE", sym_name = "foo"} : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "qux"}> ({
      %0 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
      %1 = "arith.constant"() <{value = true}> : () -> i1
      %2 = "fsm.trigger"(%1, %0) : (i1, !fsm.instancetype) -> i1
      %3 = "arith.constant"() <{value = false}> : () -> i1
      %4 = "fsm.trigger"(%3, %0) : (i1, !fsm.instancetype) -> i1
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
