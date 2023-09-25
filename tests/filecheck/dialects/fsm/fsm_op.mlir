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
  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "bar"} : () -> ()
}) : () -> ()

// -----

// CHECK:  %0 = "fsm.variable"() {initValue = 0 : i1, name = "cnt"} : () -> i1
// CHECK:  "fsm.machine"() ({
// CHECK:    "fsm.state"() ({
// CHECK:      "fsm.output"(%0) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:      }, {
// CHECK:      }) {nextState = @A} : () -> ()
// CHECK:    }) {sym_name = "A"} : () -> ()
// CHECK:    "fsm.state"() ({
// CHECK:      "fsm.output"(%0) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:      }, {
// CHECK:      }) {nextState = @C} : () -> ()
// CHECK: // CHECK:   }) {sym_name = "B"} : () -> ()
// CHECK:    "fsm.state"() ({
// CHECK:      "fsm.output"(%0) : (i1) -> ()
// CHECK:    }, {
// CHECK:      "fsm.transition"() ({
// CHECK:      }, {
// CHECK:      }) {nextState = @C} : () -> ()
// CHECK:    }) {sym_name = "C"} : () -> ()
// CHECK:  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
// CHECK:  %arg1 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK:  %arg2 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK:   %2 = "fsm.hw_instance"(%0, %arg1, %arg2) {machine = @foo, sym_name = "foo_inst"} : (i1, i16, i16) -> i1

"builtin.module"() ({

  %0 = "fsm.variable"() {initValue = 0 : i1, name = "cnt"} : () -> i1
  "fsm.machine"() ({
    "fsm.state"() ({
      "fsm.output"(%0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @A} : () -> ()
    }) {sym_name = "A"} : () -> ()
    "fsm.state"() ({
      "fsm.output"(%0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "B"} : () -> ()
    "fsm.state"() ({
      "fsm.output"(%0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
      }, {
      }) {nextState = @C} : () -> ()
    }) {sym_name = "C"} : () -> ()
  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
  %arg1 = "arith.constant"() {value = 0 : i16} : () -> i16
  %arg2 = "arith.constant"() {value = 0 : i16} : () -> i16
  %2 = "fsm.hw_instance"(%0, %arg1, %arg2) {machine = @foo, sym_name = "foo_inst"} : (i1, i16, i16) -> i1
}) : () -> ()
