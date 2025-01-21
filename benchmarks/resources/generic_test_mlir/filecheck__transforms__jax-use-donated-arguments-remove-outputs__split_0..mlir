"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {}, {tf.aliasing_output = 0 : i32}], function_type = (tensor<2x3xf32>, tensor<3x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>, sym_name = "one_donation", sym_visibility = "public"}> ({
  ^bb0(%arg6: tensor<2x3xf32>, %arg7: tensor<3x4xf32>, %arg8: tensor<2x4xf32>):
    %5 = "test.op"() : () -> tensor<2x4xf32>
    "func.return"(%5) : (tensor<2x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 0 : i32}], function_type = (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>), sym_name = "same_type_donation", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<2x3xf32>, %arg4: tensor<2x3xf32>, %arg5: tensor<2x3xf32>):
    %3 = "test.op"() : () -> tensor<2x3xf32>
    %4 = "test.op"() : () -> tensor<2x3xf32>
    "func.return"(%3, %4) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 0 : i32}, {}], function_type = (tensor<4x5xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>), sym_name = "non_trivial_donation", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<4x5xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>):
    %0 = "test.op"() : () -> tensor<2x3xf32>
    %1 = "test.op"() : () -> tensor<2x3xf32>
    %2 = "test.op"() : () -> tensor<4x5xf32>
    "func.return"(%0, %1, %2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()
