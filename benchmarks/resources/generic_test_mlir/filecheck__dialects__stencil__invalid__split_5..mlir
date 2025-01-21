"builtin.module"() ({
  "func.func"() <{function_type = (!stencil.field<[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]xf64>) -> (), sym_name = "access_bad_temp_1d"}> ({
  ^bb0(%arg0: !stencil.field<[-4,68]xf64>, %arg1: !stencil.field<[-4,68]x[-4,68]xf64>, %arg2: !stencil.field<[-4,68]xf64>):
    %0 = "stencil.load"(%arg0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %1 = "stencil.load"(%arg1) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<?x?xf64>
    %2 = "stencil.apply"(%0, %1) <{operandSegmentSizes = array<i32: 2, 0>}> ({
    ^bb0(%arg3: !stencil.temp<?xf64>, %arg4: !stencil.temp<?x?xf64>):
      %3 = "stencil.access"(%arg4) {offset = #stencil.index<[-1]>} : (!stencil.temp<?x?xf64>) -> f64
      "stencil.return"(%3) : (f64) -> ()
    }) : (!stencil.temp<?xf64>, !stencil.temp<?x?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%2, %arg2) {lb = #stencil.index<[0]>, ub = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
