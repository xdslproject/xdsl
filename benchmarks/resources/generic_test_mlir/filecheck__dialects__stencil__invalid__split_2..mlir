"builtin.module"() ({
  "func.func"() <{function_type = (!stencil.field<[-4,68]xf64>, !stencil.field<[0,1024]xf64>) -> (), sym_name = "buffer_types_mismatch_1d"}> ({
  ^bb0(%arg0: !stencil.field<[-4,68]xf64>, %arg1: !stencil.field<[0,1024]xf64>):
    %0 = "stencil.load"(%arg0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %1 = "stencil.apply"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%arg2: !stencil.temp<[-1,68]xf64>):
      %3 = "stencil.access"(%arg2) {offset = #stencil.index<[-1]>} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%3) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    %2 = "stencil.buffer"(%1) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<?xf64>
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
