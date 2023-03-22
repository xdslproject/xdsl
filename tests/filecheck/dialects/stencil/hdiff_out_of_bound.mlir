// RUN: xdsl-opt %s -t mlir -p convert-stencil-to-ll-mlir --verify-diagnostic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %3 = "stencil.cast"(%0) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %4 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %6 = "stencil.load"(%3) : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, -1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0: i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()


// CHECK-NEXT: The stencil computation requires a field with lower bound at least !stencil.index<[-1 : !i64, -1 : !i64, 0 : !i64]>, got !stencil.index<[0 : !i64, 0 : !i64, 0 : !i64]>

