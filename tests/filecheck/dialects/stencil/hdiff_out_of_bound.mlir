// RUN: xdsl-opt %s -p stencil-shape-inference --verify-diagnostic --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
    %3 = "stencil.cast"(%0) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<?x?x?xf64>) -> !stencil.field<[0,68]x[0,68]x[0,68]xf64>
    %4 = "stencil.cast"(%1) {"lb" = #stencil.index<-4, -4, -4>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %6 = "stencil.load"(%3) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<?x?x?xf64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()

// CHECK: The stencil computation requires a field with lower bound at least #stencil.index<-1, -1, 0>, got #stencil.index<0, 0, 0>, min: #stencil.index<-1, -1, 0>
