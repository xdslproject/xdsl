// RUN XDSL_ROUNDTRIP


// CHECK:  "fsm.machine"() ({
// CHECK-NEXT:  ^bb0(%arg0: i1):
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) {sym_name = "IDLE"} : () -> ()
// CHECK-NEXT:  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()

"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    "fsm.state"() ({
    }, {
    }) {sym_name = "IDLE"} : () -> ()
  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "bar"} : () -> ()
}) : () -> ()


// CHECK:  %0 = "fsm.variable"() {initValue = 0 : i1, name = "cnt"} : () -> i1
// CHECK-NEXT:  "fsm.machine"() ({
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:      "fsm.output"(%0) : (i1) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "fsm.transition"() ({
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {nextState = @A} : () -> ()
// CHECK-NEXT:    }) {sym_name = "A"} : () -> ()
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:      "fsm.output"(%0) : (i1) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "fsm.transition"() ({
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {nextState = @C} : () -> ()
// CHECK-NEXT: // CHECK-NEXT:   }) {sym_name = "B"} : () -> ()
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:      "fsm.output"(%0) : (i1) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "fsm.transition"() ({
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {nextState = @C} : () -> ()
// CHECK-NEXT:    }) {sym_name = "C"} : () -> ()
// CHECK-NEXT:  }) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()
// CHECK-NEXT:  %arg1 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK-NEXT:  %arg2 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK-NEXT:   %2 = "fsm.hw_instance"(%0, %arg1, %arg2) {machine = @foo, sym_name = "foo_inst"} : (i1, i16, i16) -> i1

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
  }) {function_type = (i16) -> (i16), initialState = "A", sym_name = "foo"} : () -> ()
  %arg1 = "arith.constant"() {value = 0 : i16} : () -> i16
  %arg2 = "arith.constant"() {value = 0 : i16} : () -> i16
  %2 = "fsm.hw_instance"(%0, %arg1, %arg2) {machine = @foo, sym_name = "foo_inst"} : (i16, i16, i16) -> i16
}) : () -> ()

// CHECK:  "builtin.module"() ({
// CHECK-NEXT:   "fsm.machine"() ({
// CHECK-NEXT:   ^bb0(%arg0: i1):
// CHECK-NEXT:     %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
// CHECK-NEXT:     "fsm.state"() ({
// CHECK-NEXT:       %1 = "arith.constant"() {value = true} : () -> i1
// CHECK-NEXT:       "fsm.output"(%arg0) : (i1) -> ()
// CHECK-NEXT:     }, {
// CHECK-NEXT:         "fsm.return"(%arg0) : (i1) -> ()
// CHECK-NEXT:       "fsm.transition"() ({
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %1 = "arith.constant"() {value = 256 : i16} : () -> i16
// CHECK-NEXT:         "fsm.update"(%0, %1) : (i16, i16) -> ()
// CHECK-NEXT:       }) {nextState = @BUSY} : () -> ()
// CHECK-NEXT:     }) {sym_name = "IDLE"} : () -> ()
// CHECK-NEXT:     "fsm.state"() ({
// CHECK-NEXT:       %1 = "arith.constant"() {value = false} : () -> i1
// CHECK-NEXT:       "fsm.output"(%arg0) : (i1) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:       "fsm.transition"() ({
// CHECK-NEXT:         %1 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK-NEXT:         %2 = "arith.cmpi"(%0, %1) {predicate = 1 : i64} : (i16, i16) -> i1
// CHECK-NEXT:         "fsm.return"(%2) : (i1) -> ()
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %1 = "arith.constant"() {value = 1 : i16} : () -> i16
// CHECK-NEXT:         %2 = "arith.subi"(%0, %1) : (i16, i16) -> i16
// CHECK-NEXT:         "fsm.update"(%0, %2) : (i16, i16) -> ()
// CHECK-NEXT:       }) {nextState = @BUSY} : () -> ()
// CHECK-NEXT:       "fsm.transition"() ({
// CHECK-NEXT:         %1 = "arith.constant"() {value = 0 : i16} : () -> i16
// CHECK-NEXT:         %2 = "arith.cmpi"(%0, %1) {predicate = 0 : i64} : (i16, i16) -> i1
// CHECK-NEXT:         "fsm.return"(%2) : (i1) -> ()
// CHECK-NEXT:       }, {
// CHECK-NEXT:       }) {nextState = @IDLE} : () -> ()
// CHECK-NEXT:     }) {sym_name = "BUSY"} : () -> ()
// CHECK-NEXT:   }) {function_type = (i1) -> (i1), initialState = "IDLE", sym_name = "foo"} : () -> ()
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:     %0 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
// CHECK-NEXT:     %1 = "arith.constant"() {value = true} : () -> i1
// CHECK-NEXT:     %2 = "fsm.trigger"(%1, %0) : (i1, !fsm.instancetype) -> i1
// CHECK-NEXT:     %3 = "arith.constant"() {value = false} : () -> i1
// CHECK-NEXT:     %4 = "fsm.trigger"(%3, %0) : (i1, !fsm.instancetype) -> i1
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {function_type = (i1) -> (i1), sym_name = "qux"} : () -> ()
// CHECK-NEXT: }) : () -> ()

"builtin.module"() ({
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      %1 = "arith.constant"() {value = true} : () -> i1
      "fsm.output"(%arg0) : (i1) -> ()
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
      "fsm.output"(%arg0) : (i1) -> ()
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
