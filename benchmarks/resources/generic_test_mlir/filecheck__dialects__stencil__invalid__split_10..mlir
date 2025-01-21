"builtin.module"() ({
  "func.func"() <{function_type = (!stencil.field<[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]xf64>) -> (), sym_name = "different_apply_bounds"}> ({
  ^bb0(%arg0: !stencil.field<[-4,68]xf64>, %arg1: !stencil.field<[-4,68]x[-4,68]xf64>, %arg2: !stencil.field<[-4,68]xf64>):
    %0 = "stencil.load"(%arg0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %1:2 = "stencil.apply"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%arg3: !stencil.temp<?xf64>):
      %2 = "stencil.access"(%arg3) {offset = #stencil.index<[-1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%2, %2) : (f64, f64) -> ()
    }) : (!stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<[0,64]xf64>)
    "stencil.store"(%1#0, %arg2) {lb = #stencil.index<[0]>, ub = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
