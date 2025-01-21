"builtin.module"() ({
  "func.func"() <{function_type = (tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>), res_attrs = [{llvm.noalias}, {llvm.noalias}], sym_name = "arg_attrs", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>):
    "func.return"(%arg0, %arg1) : (tensor<8x8xf64>, tensor<8x8xf64>) -> ()
  }) : () -> ()
}) : () -> ()
