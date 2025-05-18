// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
  "emitc.global"() <{initial_value = dense<[[2, 1], [0, 0]]> : tensor<2x2xi32>, sym_name = "input_data", type = !emitc.array<2x2xi32>}> : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "main", sym_visibility = "public"}> ({
    %0 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "emitc.get_global"() <{name = @input_data}> : () -> !emitc.array<2x2xi32>
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
