"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "noarg_void"}> ({
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "call_void"}> ({
    "func.call"() <{callee = @call_void}> : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "call_void_attributes"}> ({
    "func.call"() <{callee = @call_void_attributes}> {hello = "world"} : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> i32, sym_name = "arg_rec"}> ({
  ^bb0(%arg5: i32):
    %2 = "func.call"(%arg5) <{callee = @arg_rec}> : (i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> i32, sym_name = "arg_rec_block"}> ({
  ^bb0(%arg4: i32):
    %1 = "func.call"(%arg4) <{callee = @arg_rec_block}> : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> (i32, i32), sym_name = "external_fn", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> (i32, i32), sym_name = "multi_return_body"}> ({
  ^bb0(%arg3: i32):
    "func.return"(%arg3, %arg3) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{llvm.noalias}, {llvm.noalias}, {llvm.noalias}], function_type = (tensor<8x8xf64>, tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xf64>, sym_name = "arg_attrs", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>, %arg2: tensor<8x8xf64>):
    "func.return"(%arg0) : (tensor<8x8xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (f32, f32), res_attrs = [{dialect.a = 0 : i32}, {dialect.b = 0 : i32, dialect.c = 1 : i64}], sym_name = "output_attributes"}> ({
    %0:2 = "test.op"() : () -> (f32, f32)
    "func.return"(%0#0, %0#1) : (f32, f32) -> ()
  }) : () -> ()
}) : () -> ()
