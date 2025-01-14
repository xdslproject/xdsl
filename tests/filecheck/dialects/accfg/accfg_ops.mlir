// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

"accfg.accelerator"() <{
    name               = @acc1,
    fields             = {A=0x3c0, B=0x3c1},
    launch_fields      = {launch=0x3cf},
    barrier            = 0x7c3
}> : () -> ()

func.func @test() {
    %one, %two = "test.op"() : () -> (i32, i32)
    %zero = arith.constant 0 : i32
    %state = "accfg.setup"(%one, %two) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 0>
    }> {"test_attr" = 100 : i64} : (i32, i32) -> !accfg.state<"acc1">

    %token = "accfg.launch"(%zero, %state) <{
        param_names = ["launch"],
        accelerator = "acc1"
    }>: (i32, !accfg.state<"acc1">) -> !accfg.token<"acc1">

    %state2 = "accfg.setup"(%one, %two, %state) <{
        param_names = ["A", "B"],
        accelerator = "acc1",
        operandSegmentSizes = array<i32: 2, 1>
    }> : (i32, i32, !accfg.state<"acc1">) -> !accfg.state<"acc1">

    "accfg.await"(%token) : (!accfg.token<"acc1">) -> ()

    "test.op"() {"accfg.effects" = #accfg.effects<none>} : () -> ()

    "test.op"() {"accfg.effects" = #accfg.effects<full>} : () -> ()

    func.return
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "accfg.accelerator"() <{name = @acc1, fields = {A = 960 : i64, B = 961 : i64}, launch_fields = {launch = 975 : i64}, barrier = 1987 : i64}> : () -> ()
// CHECK-NEXT:   func.func @test() {
// CHECK-NEXT:     %one, %two = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:     %zero = arith.constant 0 : i32
// CHECK-NEXT:     %state = accfg.setup "acc1" to ("A" = %one : i32, "B" = %two : i32) attrs {test_attr = 100 : i64} : !accfg.state<"acc1">
// CHECK-NEXT:     %token = "accfg.launch"(%zero, %state) <{param_names = ["launch"], accelerator = "acc1"}> : (i32, !accfg.state<"acc1">) -> !accfg.token<"acc1">
// CHECK-NEXT:     %state2 = accfg.setup "acc1" from %state to ("A" = %one : i32, "B" = %two : i32) : !accfg.state<"acc1">
// CHECK-NEXT:     "accfg.await"(%token) : (!accfg.token<"acc1">) -> ()
// CHECK-NEXT:    "test.op"() {accfg.effects = #accfg.effects<none>} : () -> ()
// CHECK-NEXT:    "test.op"() {accfg.effects = #accfg.effects<full>} : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


// CHECK-GENERIC-NEXT: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "accfg.accelerator"() <{name = @acc1, fields = {A = 960 : i64, B = 961 : i64}, launch_fields = {launch = 975 : i64}, barrier = 1987 : i64}> : () -> ()
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "test", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:     %one, %two = "test.op"() : () -> (i32, i32)
// CHECK-GENERIC-NEXT:     %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:     %state = "accfg.setup"(%one, %two) <{param_names = ["A", "B"], accelerator = "acc1", operandSegmentSizes = array<i32: 2, 0>}>  {test_attr = 100 : i64}  : (i32, i32) -> !accfg.state<"acc1">
// CHECK-GENERIC-NEXT:     %token = "accfg.launch"(%zero, %state) <{param_names = ["launch"], accelerator = "acc1"}> : (i32, !accfg.state<"acc1">) -> !accfg.token<"acc1">
// CHECK-GENERIC-NEXT:     %state2 = "accfg.setup"(%one, %two, %state) <{param_names = ["A", "B"], accelerator = "acc1", operandSegmentSizes = array<i32: 2, 1>}> : (i32, i32, !accfg.state<"acc1">) -> !accfg.state<"acc1">
// CHECK-GENERIC-NEXT:     "accfg.await"(%token) : (!accfg.token<"acc1">) -> ()
// CHECK-GENERIC-NEXT:    "test.op"() {accfg.effects = #accfg.effects<none>} : () -> ()
// CHECK-GENERIC-NEXT:    "test.op"() {accfg.effects = #accfg.effects<full>} : () -> ()
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
