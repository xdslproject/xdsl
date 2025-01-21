"builtin.module"() ({
  "func.func"() <{function_type = (i32, f32, f64) -> (), sym_name = "test"}> ({
  ^bb0(%arg9: i32, %arg10: f32, %arg11: f64):
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{llvm.name = "preexisting_name"}, {llvm.some_other_arg_attr = "some_other_val"}, {}], function_type = (i32, f32, f64) -> (), sym_name = "test2"}> ({
  ^bb0(%arg6: i32, %arg7: f32, %arg8: f64):
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{}, {llvm.some_other_arg_attr = "some_other_val"}, {}], function_type = (i32, f32, f64) -> (), sym_name = "no_arg_names"}> ({
  ^bb0(%arg3: i32, %arg4: f32, %arg5: f64):
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32, f64) -> (), sym_name = "no_arg_attrs_or_names"}> ({
  ^bb0(%arg0: i32, %arg1: f32, %arg2: f64):
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> f64, sym_name = "decl_func", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
