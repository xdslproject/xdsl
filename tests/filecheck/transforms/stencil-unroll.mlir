// RUN: xdsl-opt %s -p "stencil-unroll{unroll-factor=8,1}" | filecheck %s

func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = arith.constant 1.0 : f64
      %6 = arith.addf %4, %5 : f64
      "stencil.return"(%6) : (f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
    func.return
  }
// CHECK:         func.func @stencil_init_float(%{{.*}} : f64, %{{.*}} : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %{{.*}} = "stencil.cast"(%{{.*}}) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.apply"(%{{.*}}) ({
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : f64):
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        "stencil.return"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"unroll" = #stencil.index<1, 8, 1>}> : (f64, f64, f64, f64, f64, f64, f64, f64) -> ()
// CHECK-NEXT:      }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
// CHECK-NEXT:      "stencil.store"(%{{.*}}, %{{.*}}) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @copy_1d(%0 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?xf64>) -> !stencil.field<[-4,68]xf64>
    %outc = "stencil.cast"(%out) : (!stencil.field<?xf64>) -> !stencil.field<[0,1024]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,68]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1>} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "stencil.store"(%3, %outc) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<[0,68]xf64>, !stencil.field<[0,1024]xf64>) -> ()
    func.return
  }

// CHECK:         func.func @copy_1d(%{{.*}} : !stencil.field<?xf64>, %{{.*}} : !stencil.field<?xf64>) {
// CHECK-NEXT:      %{{.*}} = "stencil.cast"(%{{.*}}) : (!stencil.field<?xf64>) -> !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.cast"(%{{.*}}) : (!stencil.field<?xf64>) -> !stencil.field<[0,1024]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.load"(%{{.*}}) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.apply"(%{{.*}}) ({
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stencil.temp<[-1,68]xf64>):
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1>} : (!stencil.temp<[-1,68]xf64>) -> f64
// CHECK-NEXT:        "stencil.return"(%{{.*}}) : (f64) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
// CHECK-NEXT:      "stencil.store"(%{{.*}}, %{{.*}}) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<[0,68]xf64>, !stencil.field<[0,1024]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offsets(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>, %2 : !stencil.field<?x?x?xf64>) {
    %3 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %5 = "stencil.cast"(%2) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %6 = "stencil.load"(%3) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %7, %8 = "stencil.apply"(%6) ({
    ^0(%9 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %15 = arith.addf %10, %11 : f64
      %16 = arith.addf %12, %13 : f64
      %17 = arith.addf %15, %16 : f64
      %cst = arith.constant -4.0 : f64
      %18 = arith.mulf %14, %cst : f64
      %19 = arith.addf %18, %17 : f64
      "stencil.return"(%19, %18) : (f64, f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>)
    "stencil.store"(%7, %4) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }

// CHECK:         func.func @offsets(%{{.*}} : !stencil.field<?x?x?xf64>, %{{.*}} : !stencil.field<?x?x?xf64>, %{{.*}} : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %{{.*}} = "stencil.cast"(%{{.*}}) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.cast"(%{{.*}}) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.cast"(%{{.*}}) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %{{.*}} = "stencil.load"(%{{.*}}) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:      %{{.*}}, %{{.*}} = "stencil.apply"(%{{.*}}) ({
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 2, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 2, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 2, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 3, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 2, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 3, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 3, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 4, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 2, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 3, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 4, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 4, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 5, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 3, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 4, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 5, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 5, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 6, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 4, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 5, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 6, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 6, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 7, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 5, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 6, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<-1, 7, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<1, 7, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 8, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 6, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = "stencil.access"(%{{.*}}) {"offset" = #stencil.index<0, 7, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        "stencil.return"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"unroll" = #stencil.index<1, 8, 1>}> : (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>)
// CHECK-NEXT:      "stencil.store"(%{{.*}}, %{{.*}}) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
  %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %2 = "stencil.load"(%0) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  %3 = "stencil.apply"(%2) ( {
  ^bb0(%arg2: !stencil.temp<[0,64]x[0,64]x[0,60]xf64>):  // no predecessors
    %4 = "stencil.index"() {dim = 0 : i64, offset = #stencil.index<0, 0, 0>} : () -> index
    %5 = "stencil.index"() {dim = 1 : i64, offset = #stencil.index<0, 0, 0>} : () -> index
    %6 = "stencil.index"() {dim = 2 : i64, offset = #stencil.index<0, 0, 0>} : () -> index
    %7 = "stencil.dyn_access"(%arg2, %4, %5, %6) {lb = #stencil.index<0, 0, 0>, ub = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
    %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    "stencil.return"(%8) : (!stencil.result<f64>) -> ()
  }) {lb = #stencil.index<0, 0, 0>, ub = [64, 64, 60]} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  "stencil.store"(%3, %1) {lb = #stencil.index<0, 0, 0>, ub = #stencil.index<64, 64, 60>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  return
}

// CHECK-NEXT:    func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %125 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %126 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %127 = "stencil.load"(%125) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %128 = "stencil.apply"(%127) ({
// CHECK-NEXT:      ^3(%129 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>):
// CHECK-NEXT:        %130 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 0, 0>} : () -> index
// CHECK-NEXT:        %131 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 0, 0>} : () -> index
// CHECK-NEXT:        %132 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 0, 0>} : () -> index
// CHECK-NEXT:        %133 = "stencil.dyn_access"(%129, %130, %131, %132) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %134 = "stencil.store_result"(%133) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %135 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 1, 0>} : () -> index
// CHECK-NEXT:        %136 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 1, 0>} : () -> index
// CHECK-NEXT:        %137 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 1, 0>} : () -> index
// CHECK-NEXT:        %138 = "stencil.dyn_access"(%129, %135, %136, %137) {"lb" = #stencil.index<0, 1, 0>, "ub" = #stencil.index<0, 1, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %139 = "stencil.store_result"(%138) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %140 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 2, 0>} : () -> index
// CHECK-NEXT:        %141 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 2, 0>} : () -> index
// CHECK-NEXT:        %142 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 2, 0>} : () -> index
// CHECK-NEXT:        %143 = "stencil.dyn_access"(%129, %140, %141, %142) {"lb" = #stencil.index<0, 2, 0>, "ub" = #stencil.index<0, 2, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %144 = "stencil.store_result"(%143) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %145 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 3, 0>} : () -> index
// CHECK-NEXT:        %146 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 3, 0>} : () -> index
// CHECK-NEXT:        %147 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 3, 0>} : () -> index
// CHECK-NEXT:        %148 = "stencil.dyn_access"(%129, %145, %146, %147) {"lb" = #stencil.index<0, 3, 0>, "ub" = #stencil.index<0, 3, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %149 = "stencil.store_result"(%148) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %150 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 4, 0>} : () -> index
// CHECK-NEXT:        %151 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 4, 0>} : () -> index
// CHECK-NEXT:        %152 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 4, 0>} : () -> index
// CHECK-NEXT:        %153 = "stencil.dyn_access"(%129, %150, %151, %152) {"lb" = #stencil.index<0, 4, 0>, "ub" = #stencil.index<0, 4, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %154 = "stencil.store_result"(%153) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %155 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 5, 0>} : () -> index
// CHECK-NEXT:        %156 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 5, 0>} : () -> index
// CHECK-NEXT:        %157 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 5, 0>} : () -> index
// CHECK-NEXT:        %158 = "stencil.dyn_access"(%129, %155, %156, %157) {"lb" = #stencil.index<0, 5, 0>, "ub" = #stencil.index<0, 5, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %159 = "stencil.store_result"(%158) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %160 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 6, 0>} : () -> index
// CHECK-NEXT:        %161 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 6, 0>} : () -> index
// CHECK-NEXT:        %162 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 6, 0>} : () -> index
// CHECK-NEXT:        %163 = "stencil.dyn_access"(%129, %160, %161, %162) {"lb" = #stencil.index<0, 6, 0>, "ub" = #stencil.index<0, 6, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %164 = "stencil.store_result"(%163) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        %165 = "stencil.index"() {"dim" = 0 : i64, "offset" = #stencil.index<0, 7, 0>} : () -> index
// CHECK-NEXT:        %166 = "stencil.index"() {"dim" = 1 : i64, "offset" = #stencil.index<0, 7, 0>} : () -> index
// CHECK-NEXT:        %167 = "stencil.index"() {"dim" = 2 : i64, "offset" = #stencil.index<0, 7, 0>} : () -> index
// CHECK-NEXT:        %168 = "stencil.dyn_access"(%129, %165, %166, %167) {"lb" = #stencil.index<0, 7, 0>, "ub" = #stencil.index<0, 7, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %169 = "stencil.store_result"(%168) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%134, %139, %144, %149, %154, %159, %164, %169) <{"unroll" = #stencil.index<1, 8, 1>}> : (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      "stencil.store"(%128, %126) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 60>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
