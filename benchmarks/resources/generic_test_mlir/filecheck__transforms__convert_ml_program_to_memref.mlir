"builtin.module"() ({
  "ml_program.global"() <{sym_name = "global_same_type", sym_visibility = "private", type = tensor<4xi32>, value = dense<4> : tensor<4xi32>}> : () -> ()
  %0 = "ml_program.global_load_const"() <{global = @global_same_type}> : () -> tensor<4xi32>
}) : () -> ()
