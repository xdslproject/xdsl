"builtin.module"() ({
  "func.func"() ({
  ^0(%arg0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %arg2 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %0 = "stencil.cast"(%arg0) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %1 = "stencil.cast"(%arg1) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %2 = "stencil.cast"(%arg2) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %3 = "stencil.load"(%0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[66 : i64, 66 : i64, 63 : i64]>} : (!stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>
    %4, %5, %6 = "stencil.apply"(%3) ({
    ^1(%7 : !stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>):
      %8 = "stencil.access"(%7) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %9 = "stencil.access"(%7) {"offset" = #stencil.index<[0 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %10 = "stencil.access"(%7) {"offset" = #stencil.index<[2 : i64, 2 : i64, 3 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> f64
      %11 = "arith.addf"(%9, %10) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
      %12 = "stencil.store_result"(%10) : (f64) -> !stencil.result<f64>
      %13 = "arith.subf"(%8, %11) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
      %14 = "stencil.store_result"(%13) : (f64) -> !stencil.result<f64>
      %15 = "stencil.store_result"(%11) : (f64) -> !stencil.result<f64>
      "stencil.return"(%12, %14, %15) : (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) -> ()
    }) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[66 : i64, 66 : i64, 63 : i64], f64>) -> (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>)
    "stencil.store"(%5, %1) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
    "stencil.store"(%6, %2) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
    "stencil.store"(%4, %0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 60 : i64]>} : (!stencil.temp<[64 : i64, 64 : i64, 60 : i64], f64>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "simple_stencil_inlining"} : () -> ()
}) : () -> ()


