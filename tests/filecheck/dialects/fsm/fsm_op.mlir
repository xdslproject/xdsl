// RUN XDSL_ROUNDTRIP

// CHECK:  "fsm.machine"() ({
// CHECK:  ^bb0(%arg0: i1):
// CHECK:    "fsm.state"() ({
// CHECK:    }, {
// CHECK:    }) {sym_name = "IDLE"} : () -> ()
// CHECK:  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()

"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    "fsm.state"() ({
    }, {
    }) {sym_name = "IDLE"} : () -> ()
  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()
}) : () -> ()


// CHECK:  ^bb0(%arg0: i1):
// CHECK:    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
// CHECK:    "fsm.state"() ({
// CHECK:      %1 = "arith.constant"() {value = true} : () -> i1
// CHECK:      "fsm.output"(%1) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:        "fsm.return"(%arg0) : (i1) -> ()
// CHECK:      }, {
// CHECK:        %1 = "arith.constant"() {value = 256 : i16} : () -> i16
// CHECK:        "fsm.update"(%0, %1) : (i16, i16) -> ()
// CHECK:      }) {nextState = @BUSY} : () -> ()
// CHECK:    }) {sym_name = "IDLE"} : () -> ()
// CHECK:    "fsm.state"() ({
// CHECK:      %1 = "arith.constant"() {value = false} : () -> i1
// CHECK:      "fsm.output"(%1) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:        %1 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK:        %2 = "arith.cmpi"(%0, %1) {predicate = 1 : i64} : (i16, i16) -> i1
// CHECK:        "fsm.return"(%2) : (i1) -> ()
// CHECK:      }, {
// CHECK:        %1 = "arith.constant"() {value = 1 : i16} : () -> i16
// CHECK:        %2 = "arith.subi"(%0, %1) : (i16, i16) -> i16
// CHECK:        "fsm.update"(%0, %2) : (i16, i16) -> ()
// CHECK:      }) {nextState = @BUSY} : () -> ()
// CHECK:      "fsm.transition"() ({
// CHECK:        %1 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK:        %2 = "arith.cmpi"(%0, %1) {predicate = 0 : i64} : (i16, i16) -> i1
// CHECK:        "fsm.return"(%2) : (i1) -> ()
// CHECK:      }, {
// CHECK:      }) {nextState = @IDLE} : () -> ()
// CHECK:    }) {sym_name = "BUSY"} : () -> ()
// CHECK:  }) {function_type = (i1) -> i1, initialState = "IDLE", sym_name = "foo"} : () -> ()
// CHECK:"func.func"() ({
// CHECK:    %0 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
// CHECK:    %1 = "arith.constant"() {value = true} : () -> i1
// CHECK:    %2 = "fsm.trigger"(%1, %0) : (i1, !fsm.instancetype) -> i1
// CHECK:    %3 = "arith.constant"() {value = false} : () -> i1
// CHECK:    %4 = "fsm.trigger"(%3, %0) : (i1, !fsm.instancetype) -> i1
// CHECK:    "func.return"() : () -> ()
// CHECK:  }) {function_type = () -> (), sym_name = "qux"} : () -> ()

"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      %1 = "arith.constant"() {value = true} : () -> i1
      "fsm.output"(%1) : (i1) -> ()
    }, {
      "fsm.transition"() ({
        "fsm.return"(%arg0) : (i1) -> ()
      }, {
        %1 = "arith.constant"() {value = 256 : i16} : () -> i16
        "fsm.update"(%0, %1) : (i16, i16) -> ()
      }) {nextState = @BUSY} : () -> ()
    }) {sym_name = "IDLE"} : () -> ()
    "fsm.state"() ({
      %1 = "arith.constant"() {value = false} : () -> i1
      "fsm.output"(%1) : (i1) -> ()
    }, {
      "fsm.transition"() ({
        %1 = "arith.constant"() {value = 0 : i16} : () -> i16
        %2 = "arith.cmpi"(%0, %1) {predicate = 1 : i64} : (i16, i16) -> i1
        "fsm.return"(%2) : (i1) -> ()
      }, {
        %1 = "arith.constant"() {value = 1 : i16} : () -> i16
        %2 = "arith.subi"(%0, %1) : (i16, i16) -> i16
        "fsm.update"(%0, %2) : (i16, i16) -> ()
      }) {nextState = @BUSY} : () -> ()
      "fsm.transition"() ({
        %1 = "arith.constant"() {value = 0 : i16} : () -> i16
        %2 = "arith.cmpi"(%0, %1) {predicate = 0 : i64} : (i16, i16) -> i1
        "fsm.return"(%2) : (i1) -> ()
      }, {
      }) {nextState = @IDLE} : () -> ()
    }) {sym_name = "BUSY"} : () -> ()
  }) {function_type = (i1) -> (i1), initialState = "IDLE", sym_name = "foo"} : () -> ()
  "func.func"() ({
    %0 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
    %1 = "arith.constant"() {value = true} : () -> i1
    %2 = "fsm.trigger"(%1, %0) : (i1, !fsm.instancetype) -> i1
    %3 = "arith.constant"() {value = false} : () -> i1
    %4 = "fsm.trigger"(%3, %0) : (i1, !fsm.instancetype) -> i1
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "qux"} : () -> ()
}) : () -> ()

// CHECK:  "builtin.module"() ({
// CHECK:  ^bb0(%arg0: i1, %arg1: i1):
// CHECK:    %0 = "arith.constant"() {value = true} : () -> i1
// CHECK:    %1 = "fsm.hw_instance"(%0, %arg0, %arg1) {machine = @foo, sym_name = "foo_inst"} : (i1, i1, i1) -> i1
// CHECK:  }) : () -> ()

"builtin.module"() ({
  ^bb0(%arg0: i1, %arg1: i1):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "fsm.hw_instance"(%0, %arg0, %arg1) {machine = @foo, sym_name = "foo_inst"} : (i1, i1, i1) -> i1
}) : () -> ()

// CHECK:  "fsm.machine"() ({
// CHECK:  ^bb0(%arg0: i1):
// CHECK:    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
// CHECK:    "fsm.state"() ({
// CHECK:      "fsm.output"(%arg0) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:      }, {
// CHECK:      }) {nextState = @A} : () -> ()
// CHECK:    }) {sym_name = "A"} : () -> ()
// CHECK:    "fsm.state"() ({
// CHECK:      "fsm.output"(%arg0) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:      }, {
// CHECK:      }) {nextState = @B} : () -> ()
// CHECK:    }) {sym_name = "B"} : () -> ()
// CHECK:    "fsm.state"() ({
// CHECK:      "fsm.output"(%arg0) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:      }, {
// CHECK:      }) {nextState = @C} : () -> ()
// CHECK:    }) {sym_name = "C"} : () -> ()
// CHECK:  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()

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
