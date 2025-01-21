"builtin.module"() ({
  "func.func"() <{function_type = (!stencil.field<[-4,68]xf64>) -> (), sym_name = "apply_no_return_1d"}> ({
  ^bb0(%arg0: !stencil.field<[-4,68]xf64>):
    %0 = "stencil.load"(%arg0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    "stencil.apply"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%arg1: !stencil.temp<?xf64>):
      %1 = "stencil.access"(%arg1) {offset = #stencil.index<[-1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"() : () -> ()
    }) : (!stencil.temp<?xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
