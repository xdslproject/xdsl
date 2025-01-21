"builtin.module"() ({
  "hw.module"() ({
  ^bb0(%arg0: i8):
    "hw.output"() : () -> ()
  }) {module_type = !hw.modty<input a : i32>, parameters = [], sym_name = "wrong_arg"} : () -> ()
}) : () -> ()
