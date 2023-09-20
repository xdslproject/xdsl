"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    "fsm.state"() ({
    }, {
    }) {sym_name = "IDLE"} : () -> ()
  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      %1 = "arith.constant"() <{value = true}> : () -> i1
      "fsm.output"(%1) : (i1) -> ()
    }, {
      "fsm.transition"() ({
        "fsm.return"(%arg0) : (i1) -> ()
      }, {
        %1 = "arith.constant"() <{value = 256 : i16}> : () -> i16
        "fsm.update"(%0, %1) : (i16, i16) -> ()
      }) {nextState = @BUSY} : () -> ()
    }) {sym_name = "IDLE"} : () -> ()
    "fsm.state"() ({
      %1 = "arith.constant"() <{value = false}> : () -> i1
      "fsm.output"(%1) : (i1) -> ()
    }, {
      "fsm.transition"() ({
        %1 = "arith.constant"() <{value = 0 : i16}> : () -> i16
        %2 = "arith.cmpi"(%0, %1) <{predicate = 1 : i64}> : (i16, i16) -> i1
        "fsm.return"(%2) : (i1) -> ()
      }, {
        %1 = "arith.constant"() <{value = 1 : i16}> : () -> i16
        %2 = "arith.subi"(%0, %1) : (i16, i16) -> i16
        "fsm.update"(%0, %2) : (i16, i16) -> ()
      }) {nextState = @BUSY} : () -> ()
      "fsm.transition"() ({
        %1 = "arith.constant"() <{value = 0 : i16}> : () -> i16
        %2 = "arith.cmpi"(%0, %1) <{predicate = 0 : i64}> : (i16, i16) -> i1
        "fsm.return"(%2) : (i1) -> ()
      }, {
      }) {nextState = @IDLE} : () -> ()
    }) {sym_name = "BUSY"} : () -> ()
  }) {function_type = (i1) -> i1, initialState = "IDLE", sym_name = "foo"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      "fsm.output"(%arg0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
    "fsm.state"() ({
      "fsm.output"(%arg0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @B} : () -> ()
    }) {sym_name = "B"} : () -> ()
    "fsm.state"() ({
      "fsm.output"(%arg0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "C"} : () -> ()
  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
}) : () -> ()
