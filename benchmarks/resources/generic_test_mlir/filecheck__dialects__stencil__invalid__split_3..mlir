"builtin.module"() ({
  "func.func"() <{function_type = (!stencil.temp<[0,68]xf64>) -> (), sym_name = "buffer_operand_source_1d"}> ({
  ^bb0(%arg0: !stencil.temp<[0,68]xf64>):
    %0 = "stencil.buffer"(%arg0) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
