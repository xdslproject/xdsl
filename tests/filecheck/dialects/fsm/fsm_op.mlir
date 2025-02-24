// RUN: XDSL_ROUNDTRIP

// COM: the machine consists of a single state (IDLE) with no transitions

// CHECK:  "fsm.machine"() ({
// CHECK:  ^0(%arg0 : i1):
// CHECK:    "fsm.state"() ({
// CHECK:    }, {
// CHECK:    }) {sym_name = "IDLE"} : () -> ()
// CHECK:  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()

builtin.module {
  "fsm.machine"() ({
  ^0(%arg0 : i1):
    "fsm.state"() ({
    }, {
    }) {sym_name = "IDLE"} : () -> ()
  }) {function_type = (i1) -> (), initialState = "IDLE", sym_name = "foo"} : () -> ()
}

// COM: the machine consists of three states A (initial), B, C
// COM: A outputs value resulting from fsm.variable ("cnt") and has transition A --> A
// COM: B outputs value resulting from fsm.variable ("cnt") and has transition B --> B
// COM: C outputs value resulting from fsm.variable ("cnt") and has transition C --> C
// COM: the machine is embedded in a hardware instance by HWInstanceOp

// CHECK:  %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
// CHECK-NEXT:  "fsm.machine"() ({
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:      "fsm.output"(%0) : (i16) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "fsm.transition"() ({
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {nextState = @A} : () -> ()
// CHECK-NEXT:    }) {sym_name = "A"} : () -> ()
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:      "fsm.output"(%0) : (i16) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "fsm.transition"() ({
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {nextState = @C} : () -> ()
// CHECK-NEXT:   }) {sym_name = "B"} : () -> ()
// CHECK-NEXT:    "fsm.state"() ({
// CHECK-NEXT:      "fsm.output"(%0) : (i16) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "fsm.transition"() ({
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {nextState = @C} : () -> ()
// CHECK-NEXT:    }) {sym_name = "C"} : () -> ()
// CHECK-NEXT:  }) {function_type = (i16) -> i16, initialState = "A", sym_name = "foo"} : () -> ()
// CHECK-NEXT:  %arg1 = arith.constant 0 : i16
// CHECK-NEXT:  %arg2 = arith.constant 50 : i16
// CHECK-NEXT:  %1 = "fsm.hw_instance"(%0, %arg1, %arg2) {machine = @foo, sym_name = "foo_inst"} : (i16, i16, i16) -> i16

builtin.module {

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
  %arg1 = arith.constant 0 : i16
  %arg2 = arith.constant 50 : i16
  %2 = "fsm.hw_instance"(%0, %arg1, %arg2) {machine = @foo, sym_name = "foo_inst"} : (i16, i16, i16) -> i16
}

// COM: the machine consists of two states
// COM: IDLE outputs constant value true
// COM: BUSY outputs constant value false
// COM: and the following transitions:
// COM: IDLE --> BUSY returns a value from the guard region (fsm.return) and updates a variable (fsm.update)
// COM: BUSY --> IDLE returns a value from the guard region (fsm.return) resulting from arithmetic operations
// COM: BUSY --> BUSY returns a value resulting from arithmetic operations from the guard region and updates %0
// COM: with %2 according to the result of previous arith operations
// COM: the machine is embedded in a SW instance by InstanceOp and TriggerOp

// CHECK:      builtin.module {
// CHECK-NEXT:   "fsm.machine"() ({
// CHECK-NEXT:   ^0(%arg0 : i1):
// CHECK-NEXT:     %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
// CHECK-NEXT:     "fsm.state"() ({
// CHECK-NEXT:       %1 = arith.constant true
// CHECK-NEXT:       "fsm.output"(%arg0) : (i1) -> ()
// CHECK-NEXT:     }, {
// CHECK-NEXT:       "fsm.transition"() ({
// CHECK-NEXT:         "fsm.return"(%arg0) : (i1) -> ()
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %2 = arith.constant 256 : i16
// CHECK-NEXT:         "fsm.update"(%0, %2) : (i16, i16) -> ()
// CHECK-NEXT:       }) {nextState = @BUSY} : () -> ()
// CHECK-NEXT:     }) {sym_name = "IDLE"} : () -> ()
// CHECK-NEXT:     "fsm.state"() ({
// CHECK-NEXT:       %3 = arith.constant false
// CHECK-NEXT:       "fsm.output"(%arg0) : (i1) -> ()
// CHECK-NEXT:     }, {
// CHECK-NEXT:       "fsm.transition"() ({
// CHECK-NEXT:         %4 = arith.constant 0 : i16
// CHECK-NEXT:         %5 = arith.cmpi ne, %0, %4 : i16
// CHECK-NEXT:         "fsm.return"(%5) : (i1) -> ()
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %6 = arith.constant 1 : i16
// CHECK-NEXT:         %7 = arith.subi %0, %6 : i16
// CHECK-NEXT:         "fsm.update"(%0, %7) : (i16, i16) -> ()
// CHECK-NEXT:       }) {nextState = @BUSY} : () -> ()
// CHECK-NEXT:       "fsm.transition"() ({
// CHECK-NEXT:         %8 = arith.constant 0 : i16
// CHECK-NEXT:         %9 = arith.cmpi eq, %0, %8 : i16
// CHECK-NEXT:         "fsm.return"(%9) : (i1) -> ()
// CHECK-NEXT:       }, {
// CHECK-NEXT:       }) {nextState = @IDLE} : () -> ()
// CHECK-NEXT:     }) {sym_name = "BUSY"} : () -> ()
// CHECK-NEXT:   }) {function_type = (i1) -> i1, initialState = "IDLE", sym_name = "foo"} : () -> ()
// CHECK-NEXT:   func.func @qux() {
// CHECK-NEXT:     %10 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
// CHECK-NEXT:     %11 = arith.constant true
// CHECK-NEXT:     %12 = "fsm.trigger"(%11, %10) : (i1, !fsm.instancetype) -> i1
// CHECK-NEXT:     %13 = arith.constant false
// CHECK-NEXT:     %14 = "fsm.trigger"(%13, %10) : (i1, !fsm.instancetype) -> i1
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


builtin.module {
  "fsm.machine"() ({
  ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
      %1 = arith.constant true
      "fsm.output"(%arg0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
        "fsm.return"(%arg0) : (i1) -> ()
      }, {
        %1 = arith.constant 256 : i16
        "fsm.update"(%0, %1) : (i16, i16) -> ()
      }) {nextState = @BUSY} : () -> ()
    }) {sym_name = "IDLE"} : () -> ()
    "fsm.state"() ({
      %1 = arith.constant false
      "fsm.output"(%arg0) : (i1) -> ()
    }, {
      "fsm.transition"() ({
        %1 = arith.constant 0 : i16
        %2 = "arith.cmpi"(%0, %1) {predicate = 1 : i64} : (i16, i16) -> i1
        "fsm.return"(%2) : (i1) -> ()
      }, {
        %1 = arith.constant 1 : i16
        %2 = "arith.subi"(%0, %1) : (i16, i16) -> i16
        "fsm.update"(%0, %2) : (i16, i16) -> ()
      }) {nextState = @BUSY} : () -> ()
      "fsm.transition"() ({
        %1 = arith.constant 0 : i16
        %2 = "arith.cmpi"(%0, %1) {predicate = 0 : i64} : (i16, i16) -> i1
        "fsm.return"(%2) : (i1) -> ()
      }, {
      }) {nextState = @IDLE} : () -> ()
    }) {sym_name = "BUSY"} : () -> ()
  }) {function_type = (i1) -> (i1), initialState = "IDLE", sym_name = "foo"} : () -> ()
  "func.func"() ({
    %0 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
    %1 = arith.constant true
    %2 = "fsm.trigger"(%1, %0) : (i1, !fsm.instancetype) -> i1
    %3 = arith.constant false
    %4 = "fsm.trigger"(%3, %0) : (i1, !fsm.instancetype) -> i1
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "qux"} : () -> ()
}
