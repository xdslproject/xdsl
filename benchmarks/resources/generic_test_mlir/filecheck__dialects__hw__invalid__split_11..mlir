"builtin.module"() ({
  "hw.module"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    "hw.output"() : () -> ()
  }) {module_type = !hw.modty<input a : i32>, parameters = [], sym_name = "too_many_args"} : () -> ()
}) : () -> ()
