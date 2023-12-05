func.func @stencil(%ibuff : !stencil.field<[0,128]xf64>, %obuff : !stencil.field<[0,128]xf64>) {
  %source = "stencil.load"(%ibuff) : (!stencil.field<[0,128]xf64>)
                               -> !stencil.temp<?xf64>
  %out = "stencil.apply"(%source) ({
    ^bb(%arg : !stencil.temp<?xf64>):
    %l = "stencil.access"(%arg) {"offset" = !stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
    %c = "stencil.access"(%arg) {"offset" = !stencil.index<0>} : (!stencil.temp<?xf64>) -> f64
    %r = "stencil.access"(%arg) {"offset" = !stencil.index<1>} : (!stencil.temp<?xf64>) -> f64
    %2 = arith.constant -2.0 : f64
    %c2 = arith.mulf %c, %2 : f64
    %s1 = arith.addf %l, %r : f64
    %v = arith.addf %s1, %c2 : f64
    // %v = %l + %r - 2.0 * %c
    "stencil.return"(%v) : (f64) -> ()
  }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64> 
  
  "stencil.store"(%out, %obuff) {"lb" = #stencil.index<1>, "ub" = #stencil.index<127>} : (!stencil.temp<?xf64>, !stencil.field<[0,128]xf64>) -> ()
  func.return
}