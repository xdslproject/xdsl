// xDSL's laplace implementation.

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg1: !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):  // no predecessors
    %0 = "stencil.cast"(%arg0) {"lb" = !stencil.index<[-4, -4, -4]>, "ub" = !stencil.index<[68, 68, 68]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %1 = "stencil.cast"(%arg1) {"lb" = !stencil.index<[-4, -4, -4]>, "ub" = !stencil.index<[68, 68, 68]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %2 = "stencil.load"(%0) : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    %3 = "stencil.apply"(%2) ( {
    ^bb0(%arg2: !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>):  // no predecessors
      %4 = "stencil.access"(%arg2) {"offset" = !stencil.index<[-1, 0, 0]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %5 = "stencil.access"(%arg2) {"offset" = !stencil.index<[1, 0, 0]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %6 = "stencil.access"(%arg2) {"offset" = !stencil.index<[0, 1, 0]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %7 = "stencil.access"(%arg2) {"offset" = !stencil.index<[0, -1, 0]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %8 = "stencil.access"(%arg2) {"offset" = !stencil.index<[0, 0, 0]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %9 = "arith.addf"(%4, %5) : (f64, f64) -> f64
      %10 = "arith.addf"(%6, %7) : (f64, f64) -> f64
      %11 = "arith.addf"(%9, %10) : (f64, f64) -> f64
      %cst = "arith.constant"() {value = -4.000000e+00 : f64} : () -> f64
      %12 = "arith.mulf"(%8, %cst) : (f64, f64) -> f64
      %13 = "arith.addf"(%12, %11) : (f64, f64) -> f64
      "stencil.return"(%13) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    "stencil.store"(%3, %1) {"lb" = !stencil.index<[0, 0, 0]>, "ub" = !stencil.index<[64, 64, 64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {stencil.program, sym_name = "laplace_xdsl", function_type = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> ()} : () -> ()
}) : () -> ()
