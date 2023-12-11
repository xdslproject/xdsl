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
