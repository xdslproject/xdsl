"builtin.module"() ({
  "func.func"() <{function_type = (!stencil.field<[-4,68]xf64>) -> (), sym_name = "access_out_of_apply_1d"}> ({
  ^bb0(%arg0: !stencil.field<[-4,68]xf64>):
    %0 = "stencil.load"(%arg0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %1 = "stencil.access"(%0) {offset = #stencil.index<[0]>} : (!stencil.temp<?xf64>) -> f64
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
