// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

"fsm.machine"() ({
    ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
        "fsm.output"(%arg0) : (i1) -> ()
    }, {
        "fsm.transition"() ({
        }, {
        }) {nextState = @B} : () -> ()
    }) {sym_name = "B"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()

// CHECK: Can not find initial state

// -----

"fsm.machine"() ({
    ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
        "fsm.output"(%arg0) : (i1) -> ()
    }, {
        "fsm.transition"() ({
        }, {
        }) {nextState = @C} : () -> ()
    }) {sym_name = "B"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "B", sym_name = "foo"} : () -> ()

// CHECK: Can not find next state

// -----

"fsm.machine"() ({
^bb0(%arg0: i1):
%0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
"fsm.state"() ({
    "fsm.output"(%arg0) : (i1) -> ()
}, {
    "fsm.transition"() ({
    ^bb1(%arg1: i1): "fsm.update"(%arg1, %arg1) : (i1, i1) -> ()
    }, {
    }) {nextState = @A} : () -> ()
}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()

// CHECK: Destination is not a variable operation

// -----

"fsm.machine"() ({
^bb0(%arg0: i1):
%0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
"fsm.state"() ({
    "fsm.output"(%arg0) : (i1) -> ()
}, {
    "fsm.transition"() ({
    ^bb1(%arg1: i1): "test.termop"() : () -> ()
    }, {
    }) {nextState = @A} : () -> ()
}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()

// CHECK: Guard region must terminate with ReturnOp

// -----

"fsm.machine"() ({
^bb0(%arg0: i1):
%0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
"fsm.state"() ({
    ^bb1(%arg1: i1): "fsm.transition"() ({
        ^bb2(%arg2: i2): "fsm.return"() : () -> ()
        }, {
    }) {nextState = @A} : () -> ()
},{}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()

// CHECK: Transition must be located in a transitions region

// -----

"fsm.machine"() ({
    ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
        "fsm.output"(%arg0) : (i1) -> ()
    }, {
    }) {sym_name = "B"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "B", sym_name = "foo", arg_names = ["argument"] } : () -> ()

// CHECK: arg_attrs must be consistent with arg_names

// -----

"fsm.machine"() ({
    ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
        "fsm.output"(%arg0) : (i1) -> ()
    }, {
    }) {sym_name = "B"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "B", sym_name = "foo", arg_attrs = [{"name"="1","type"="2"}] } : () -> ()

// CHECK: arg_attrs must be consistent with arg_names

// -----

"fsm.machine"() ({
    ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
        "fsm.output"(%arg0) : (i1) -> ()
    }, {
    }) {sym_name = "B"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "B", sym_name = "foo", arg_names = ["names","of"],arg_attrs = [{"name"="1","type"="2"}] } : () -> ()

// CHECK: The number of arg_attrs and arg_names should be the same"

// -----

"fsm.machine"() ({
    ^bb0(%arg0: i1):
    %0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
    "fsm.state"() ({
        "fsm.output"(%arg0) : (i1) -> ()
    }, {
    }) {sym_name = "B"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "B", sym_name = "foo", res_names = ["names"],res_attrs = [{"name"="1","type"="2"}, {"name"="3","type"="4"}] } : () -> ()

// CHECK: The number of res_attrs and res_names should be the same"

// -----

"fsm.machine"() ({
^bb0(%arg0: i1):
%0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
"fsm.state"() ({
}, {
    "fsm.transition"() ({
    }, {
    }) {nextState = @A} : () -> ()
    "fsm.output"(%arg0) : (i1) -> ()
}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo"} : () -> ()

// CHECK: Transition regions should not output any value

// -----

"fsm.machine"() ({
^bb0(%arg0: i1):
%0 = "fsm.variable"() {initValue = 0 : i16, name = "cnt"} : () -> i16
"fsm.state"() ({
    ^bb1(%arg1: i1):
}, {
    "fsm.transition"() ({
        "fsm.return"(%arg0) : (i1) -> ()
    }, {
        
    }) {nextState = @A} : () -> ()
    
}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo", res_names = ["names"],res_attrs = [{"name"="1","type"="2"}] } : () -> ()

// CHECK: State must have a non-empty output region when the machine has results

// -----
"fsm.machine"() ({

^bb0(%arg0: i1):
%arg1 = "fsm.variable"() {initValue = 0 : i16, name = "v1"} : () -> i16
%arg2 = "fsm.variable"() {initValue = 2 : i16, name = "v2"} : () -> i16

"fsm.state"() ({
    "fsm.output"(%arg0) : (i1) -> ()
    
}, {
    "fsm.transition"() ({
        ^bb1(%arg3: i1): 
            "fsm.update"(%arg1, %arg2) {variable = "v1" , value = "v2"}: (i16,i16) -> ()
    }, {
        
    }) {nextState = @A} : () -> ()
    
}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo", res_names = ["names"],res_attrs = [{"name"="1","type"="2"}] } : () -> ()

// Update must only be located in the action region of a transition

// -----
"fsm.machine"() ({

^bb0(%arg0: i1):
%arg1 = "fsm.variable"() {initValue = 0 : i16, name = "v1"} : () -> i16
%arg2 = "fsm.variable"() {initValue = 2 : i16, name = "v2"} : () -> i16

"fsm.state"() ({
    "fsm.output"(%arg0) : (i1) -> ()
    
}, {
    "fsm.transition"() ({
        
    }, {
        ^bb1(%arg3: i1): 
            "fsm.update"(%arg1, %arg2) {variable = "v1" , value = "v2"}: (i16,i16) -> ()
            "fsm.update"(%arg1, %arg2) {variable = "v1" , value = "v2"}: (i16,i16) -> ()
    }) {nextState = @A} : () -> ()
    
}) {sym_name = "A"} : () -> ()
}) {function_type = (i1) -> i1, initialState = "A", sym_name = "foo", res_names = ["names"],res_attrs = [{"name"="1","type"="2"}] } : () -> ()

// Multiple updates to the same variable within a single action region is disallowed

// -----

"fsm.machine"() ({

"fsm.state"() ({
    ^bb2(): "fsm.output"() : () -> ()

}, {
    "fsm.transition"() ({
        
    }, {
    }) {nextState = @A} : () -> ()
    
}) {sym_name = "A"} : () -> ()

}) {function_type = (i16) -> ([i16]) , initialState = "A", sym_name = "foo"} : () -> ()

// Output types must be consistent with the machine's