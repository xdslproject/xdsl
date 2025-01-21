"builtin.module"() ({
  "func.func"() <{function_type = (tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>), res_attrs = [{llvm.noalias}, {llvm.noalias}], sym_name = "arg_attrs", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>):
    "func.return"(%arg0, %arg1) : (tensor<8x8xf64>, tensor<8x8xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (f32, f32), res_attrs = [{dialect.a = 0 : i32}, {dialect.b = 0 : i32, dialect.c = 1 : i64}], sym_name = "output_attributes"}> ({
    %0:2 = "test.op"() : () -> (f32, f32)
    "func.return"(%0#0, %0#1) : (f32, f32) -> ()
  }) : () -> ()
}) : () -> ()
