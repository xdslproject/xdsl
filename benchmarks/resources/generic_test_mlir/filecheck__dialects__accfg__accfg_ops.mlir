"builtin.module"() ({
  "accfg.accelerator"() <{barrier = 1987 : i64, fields = {A = 960 : i64, B = 961 : i64}, launch_fields = {launch = 975 : i64}, name = @acc1}> : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "test"}> ({
    %0:2 = "test.op"() : () -> (i32, i32)
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "accfg.setup"(%0#0, %0#1) <{accelerator = "acc1", operandSegmentSizes = array<i32: 2, 0>, param_names = ["A", "B"]}> {test_attr = 100 : i64} : (i32, i32) -> !accfg.state<"acc1">
    %3 = "accfg.launch"(%1, %2) <{accelerator = "acc1", param_names = ["launch"]}> : (i32, !accfg.state<"acc1">) -> !accfg.token<"acc1">
    %4 = "accfg.setup"(%0#0, %0#1, %2) <{accelerator = "acc1", operandSegmentSizes = array<i32: 2, 1>, param_names = ["A", "B"]}> : (i32, i32, !accfg.state<"acc1">) -> !accfg.state<"acc1">
    "accfg.await"(%3) : (!accfg.token<"acc1">) -> ()
    "test.op"() {accfg.effects = #accfg.effects<none>} : () -> ()
    "test.op"() {accfg.effects = #accfg.effects<full>} : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
