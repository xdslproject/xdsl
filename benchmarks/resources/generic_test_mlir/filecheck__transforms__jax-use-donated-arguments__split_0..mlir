"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {}, {tf.aliasing_output = 0 : i32}], function_type = (tensor<2x3xf32>, tensor<3x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>, sym_name = "one_donation", sym_visibility = "public"}> ({
  ^bb0(%arg9: tensor<2x3xf32>, %arg10: tensor<3x4xf32>, %arg11: tensor<2x4xf32>):
    %9 = "test.op"() : () -> tensor<2x4xf32>
    "func.return"(%9) : (tensor<2x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 0 : i32}], function_type = (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>), sym_name = "same_type_donation", sym_visibility = "public"}> ({
  ^bb0(%arg6: tensor<2x3xf32>, %arg7: tensor<2x3xf32>, %arg8: tensor<2x3xf32>):
    %7 = "test.op"() : () -> tensor<2x3xf32>
    %8 = "test.op"() : () -> tensor<2x3xf32>
    "func.return"(%7, %8) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 0 : i32}, {}], function_type = (tensor<4x5xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>), sym_name = "non_trivial_donation", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<4x5xf32>, %arg4: tensor<2x3xf32>, %arg5: tensor<2x3xf32>):
    %4 = "test.op"() : () -> tensor<2x3xf32>
    %5 = "test.op"() : () -> tensor<2x3xf32>
    %6 = "test.op"() : () -> tensor<4x5xf32>
    "func.return"(%4, %5, %6) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{}, {}, {tf.aliasing_output = 0 : i32}], function_type = (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>), sym_name = "dont_double_buffer", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>):
    %0 = "test.op"() : () -> tensor<2x3xf32>
    %1 = "test.op"() : () -> tensor<2x3xf32>
    %2 = "bufferization.materialize_in_destination"(%0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = "bufferization.materialize_in_destination"(%1, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    "func.return"(%2, %3) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()
