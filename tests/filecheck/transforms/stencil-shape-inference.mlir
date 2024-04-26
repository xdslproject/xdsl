// RUN: xdsl-opt -p stencil-shape-inference --verify-diagnostics --split-input-file %s | filecheck %s

builtin.module {
  func.func @different_input_offsets(%out : !stencil.field<[-4,68]xf64>, %left : !stencil.field<[-4,68]xf64>, %right : !stencil.field<[-4,68]xf64>) {
    %tleft = "stencil.load"(%left) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %tright = "stencil.load"(%right) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %tout = "stencil.apply"(%tleft, %tright) ({
    ^0(%lefti : !stencil.temp<?xf64>, %righti : !stencil.temp<?xf64>):
      %l = "stencil.access"(%lefti) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
      %r = "stencil.access"(%righti) {"offset" = #stencil.index< 1>} : (!stencil.temp<?xf64>) -> f64
      %o = arith.addf %l, %r : f64
      "stencil.return"(%o) : (f64) -> ()
    }) : (!stencil.temp<?xf64>, !stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%tout, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @different_input_offsets(%out : !stencil.field<[-4,68]xf64>, %left : !stencil.field<[-4,68]xf64>, %right : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:     %tleft = "stencil.load"(%left) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:     %tright = "stencil.load"(%right) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[1,65]xf64>
// CHECK-NEXT:     %tout = "stencil.apply"(%tleft, %tright) ({
// CHECK-NEXT:     ^0(%lefti : !stencil.temp<[-1,63]xf64>, %righti : !stencil.temp<[1,65]xf64>):
// CHECK-NEXT:       %l = "stencil.access"(%lefti) {"offset" = #stencil.index<-1>} : (!stencil.temp<[-1,63]xf64>) -> f64
// CHECK-NEXT:       %r = "stencil.access"(%righti) {"offset" = #stencil.index<1>} : (!stencil.temp<[1,65]xf64>) -> f64
// CHECK-NEXT:       %o = arith.addf %l, %r : f64
// CHECK-NEXT:       "stencil.return"(%o) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,63]xf64>, !stencil.temp<[1,65]xf64>) -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%tout, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<[0,64]xf64>, !stencil.field<[-4,68]xf64>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = arith.constant -4.0 : f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      "stencil.return"(%16) : (f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }
}


// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^0(%6 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %12 = arith.addf %7, %8 : f64
// CHECK-NEXT:       %13 = arith.addf %9, %10 : f64
// CHECK-NEXT:       %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:       %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:       %15 = arith.mulf %11, %cst : f64
// CHECK-NEXT:       %16 = arith.addf %15, %14 : f64
// CHECK-NEXT:       "stencil.return"(%16) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[0,68]x[0,68]x[0,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[0,68]x[0,68]x[0,68]xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<?x?x?xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = arith.constant -4.0 : f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      "stencil.return"(%16) : (f64) -> ()
    }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }
}

// CHECK: The stencil computation requires a field with lower bound at least #stencil.index<-1, -1, 0>, got #stencil.index<0, 0, 0>, min: #stencil.index<-1, -1, 0>

// -----

builtin.module {

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
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:     %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:     %3 = "stencil.apply"(%0) ({
// CHECK-NEXT:     ^0(%4 : f64):
// CHECK-NEXT:       %5 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:       %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:       "stencil.return"(%6) : (f64) -> ()
// CHECK-NEXT:     }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
// CHECK-NEXT:     "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func private @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
    %4 = "stencil.load"(%0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %5 = "stencil.apply"(%4) ({
    ^0(%6 : !stencil.temp<?xf64>):
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%11) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>

    %15 = "stencil.buffer"(%5) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>

    %12 = "stencil.apply"(%15) ({
    ^0(%13 : !stencil.temp<?xf64>):
      %14 = "stencil.access"(%13) {"offset" = #stencil.index<0>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%14) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%12, %1) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:     %2 = "stencil.load"(%0) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:     %3 = "stencil.apply"(%2) ({
// CHECK-NEXT:     ^0(%4 : !stencil.temp<[0,64]xf64>):
// CHECK-NEXT:       %5 = "stencil.access"(%4) {"offset" = #stencil.index<0>} : (!stencil.temp<[0,64]xf64>) -> f64
// CHECK-NEXT:       "stencil.return"(%5) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:     %6 = "stencil.buffer"(%3) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:     %7 = "stencil.apply"(%6) ({
// CHECK-NEXT:     ^1(%8 : !stencil.temp<[0,64]xf64>):
// CHECK-NEXT:       %9 = "stencil.access"(%8) {"offset" = #stencil.index<0>} : (!stencil.temp<[0,64]xf64>) -> f64
// CHECK-NEXT:       "stencil.return"(%9) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%7, %1) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<[0,64]xf64>, !stencil.field<[-4,68]xf64>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func @dyn_access(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %2 = "stencil.load"(%0) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<?x?x?xf64>
  %3 = "stencil.apply"(%2) ( {
  ^bb0(%arg2: !stencil.temp<?x?x?xf64>):
    %c0 = arith.constant 0 : index
    %4 = "stencil.dyn_access"(%arg2, %c0, %c0, %c0) {lb = #stencil.index<-1, -2, 0>, ub = #stencil.index<1, 2, 0>} : (!stencil.temp<?x?x?xf64>, index, index, index) -> f64
    %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    "stencil.return"(%5) : (!stencil.result<f64>) -> ()
  }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  "stencil.store"(%3, %1) {lb = #stencil.index<0, 0, 0>, ub = #stencil.index<64, 64, 60>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = "stencil.load"(%0) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>
// CHECK-NEXT:      %3 = "stencil.apply"(%2) ({
// CHECK-NEXT:      ^0(%arg2 : !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>):
// CHECK-NEXT:        %c0 = arith.constant 0 : index
// CHECK-NEXT:        %4 = "stencil.dyn_access"(%arg2, %c0, %c0, %c0) {"lb" = #stencil.index<-1, -2, 0>, "ub" = #stencil.index<1, 2, 0>} : (!stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>, index, index, index) -> f64
// CHECK-NEXT:        %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%5) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      "stencil.store"(%3, %1) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 60>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func @combine(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %2 = "stencil.load"(%0) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<?x?x?xf64>
  %3 = "stencil.apply"(%2) ( {
  ^bb0(%arg2: !stencil.temp<?x?x?xf64>):
    %6 = "stencil.access"(%arg2) {offset = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
    "stencil.return"(%7) : (!stencil.result<f64>) -> ()
  }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %4 = "stencil.apply"(%2) ( {
  ^bb0(%arg2: !stencil.temp<?x?x?xf64>):
    %6 = "stencil.access"(%arg2) {offset = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = "stencil.store_result"(%6) : (f64) -> !stencil.result<f64>
    "stencil.return"(%7) : (!stencil.result<f64>) -> ()
  }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %5 = "stencil.combine"(%3, %4) {dim = 0 : index, index = 32 : index, operandSegmentSizes = array<i32:1, 1, 0, 0>} : (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  "stencil.store"(%5, %1) {lb = #stencil.index<0, 0, 0>, ub = #stencil.index<64, 64, 60>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @combine(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = "stencil.load"(%0) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<[0,32]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %3 = "stencil.apply"(%2) ({
// CHECK-NEXT:      ^0(%arg2 : !stencil.temp<[0,32]x[0,64]x[0,60]xf64>):
// CHECK-NEXT:        %4 = "stencil.access"(%arg2) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,32]x[0,64]x[0,60]xf64>) -> f64
// CHECK-NEXT:        %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%5) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[0,32]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,32]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %6 = "stencil.apply"(%2) ({
// CHECK-NEXT:      ^1(%arg2_1 : !stencil.temp<[32,64]x[0,64]x[0,60]xf64>):
// CHECK-NEXT:        %7 = "stencil.access"(%arg2_1) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[32,64]x[0,64]x[0,60]xf64>) -> f64
// CHECK-NEXT:        %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%8) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[0,32]x[0,64]x[0,60]xf64>) -> !stencil.temp<[32,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %9 = "stencil.combine"(%3, %6) {"dim" = 0 : index, "index" = 32 : index, "operandSegmentSizes" = array<i32: 1, 1, 0, 0>} : (!stencil.temp<[0,32]x[0,64]x[0,60]xf64>, !stencil.temp<[32,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      "stencil.store"(%9, %1) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 60>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func @buffer(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %14 = "stencil.cast"(%arg2) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %2 = "stencil.apply"() ( {
    %cst = arith.constant 1.0 : f64
    %9 = "stencil.store_result"(%cst) : (f64) -> !stencil.result<f64>
    "stencil.return"(%9) : (!stencil.result<f64>) -> ()
  }) : () -> !stencil.temp<?x?x?xf64>
  %11 = "stencil.apply"() ( {
    %cst = arith.constant 1.0 : f64
    %9 = "stencil.store_result"(%cst) : (f64) -> !stencil.result<f64>
    "stencil.return"(%9) : (!stencil.result<f64>) -> ()
  }) : () -> !stencil.temp<?x?x?xf64>
  %3 = "stencil.apply"() ( {
    %cst = arith.constant 1.0 : f64
    %9 = "stencil.store_result"(%cst) : (f64) -> !stencil.result<f64>
    "stencil.return"(%9) : (!stencil.result<f64>) -> ()
  }) : () -> !stencil.temp<?x?x?xf64>
  %4:3 = "stencil.combine"(%2, %2, %3, %11) {dim = 0 : index, index = 32 : index, operandSegmentSizes = array<i32:1, 1, 1, 1>} : (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>)
  %5 = "stencil.buffer"(%4#0) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %12 = "stencil.buffer"(%4#1) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %6 = "stencil.buffer"(%4#2) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %7 = "stencil.apply"(%5) ( {
  ^bb0(%arg3: !stencil.temp<?x?x?xf64>):  // no predecessors
    %9 = "stencil.access"(%arg3) {offset = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
    %10 = "stencil.store_result"(%9) : (f64) -> !stencil.result<f64>
    "stencil.return"(%10) : (!stencil.result<f64>) -> ()
  }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %13 = "stencil.apply"(%12) ( {
  ^bb0(%arg3: !stencil.temp<?x?x?xf64>):  // no predecessors
    %9 = "stencil.access"(%arg3) {offset = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
    %10 = "stencil.store_result"(%9) : (f64) -> !stencil.result<f64>
    "stencil.return"(%10) : (!stencil.result<f64>) -> ()
  }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %8 = "stencil.apply"(%6) ( {
  ^bb0(%arg3: !stencil.temp<?x?x?xf64>):  // no predecessors
    %9 = "stencil.access"(%arg3) {offset = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
    %10 = "stencil.store_result"(%9) : (f64) -> !stencil.result<f64>
    "stencil.return"(%10) : (!stencil.result<f64>) -> ()
  }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  "stencil.store"(%7, %0) {lb = #stencil.index<0, 0, 0>, ub = #stencil.index<64, 64, 60>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  "stencil.store"(%13, %1) {lb = #stencil.index<0, 0, 0>, ub = #stencil.index<16, 64, 60>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  "stencil.store"(%8, %14) {lb = #stencil.index<48, 0, 0>, ub = #stencil.index<64, 64, 60>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @buffer(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = "stencil.cast"(%arg2) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %3 = "stencil.apply"() ({
// CHECK-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %4 = "stencil.store_result"(%cst) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%4) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : () -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %5 = "stencil.apply"() ({
// CHECK-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %6 = "stencil.store_result"(%cst_1) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%6) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : () -> !stencil.temp<[32,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %7 = "stencil.apply"() ({
// CHECK-NEXT:        %cst_2 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %8 = "stencil.store_result"(%cst_2) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%8) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : () -> !stencil.temp<[0,32]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %9, %10, %11 = "stencil.combine"(%3, %3, %7, %5) {"dim" = 0 : index, "index" = 32 : index, "operandSegmentSizes" = array<i32: 1, 1, 1, 1>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.temp<[0,32]x[0,64]x[0,60]xf64>, !stencil.temp<[32,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.temp<[0,16]x[0,64]x[0,60]xf64>, !stencil.temp<[48,64]x[0,64]x[0,60]xf64>)
// CHECK-NEXT:      %12 = "stencil.buffer"(%9) : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %13 = "stencil.buffer"(%10) : (!stencil.temp<[0,16]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,16]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %14 = "stencil.buffer"(%11) : (!stencil.temp<[48,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[48,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %15 = "stencil.apply"(%12) ({
// CHECK-NEXT:      ^0(%arg3 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>):
// CHECK-NEXT:        %16 = "stencil.access"(%arg3) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> f64
// CHECK-NEXT:        %17 = "stencil.store_result"(%16) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%17) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %18 = "stencil.apply"(%13) ({
// CHECK-NEXT:      ^1(%arg3_1 : !stencil.temp<[0,16]x[0,64]x[0,60]xf64>):
// CHECK-NEXT:        %19 = "stencil.access"(%arg3_1) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,16]x[0,64]x[0,60]xf64>) -> f64
// CHECK-NEXT:        %20 = "stencil.store_result"(%19) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%20) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[0,16]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,16]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %21 = "stencil.apply"(%14) ({
// CHECK-NEXT:      ^2(%arg3_2 : !stencil.temp<[48,64]x[0,64]x[0,60]xf64>):
// CHECK-NEXT:        %22 = "stencil.access"(%arg3_2) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[48,64]x[0,64]x[0,60]xf64>) -> f64
// CHECK-NEXT:        %23 = "stencil.store_result"(%22) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:        "stencil.return"(%23) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[48,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[48,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      "stencil.store"(%15, %0) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 60>} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
// CHECK-NEXT:      "stencil.store"(%18, %1) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<16, 64, 60>} : (!stencil.temp<[0,16]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
// CHECK-NEXT:      "stencil.store"(%21, %2) {"lb" = #stencil.index<48, 0, 0>, "ub" = #stencil.index<64, 64, 60>} : (!stencil.temp<[48,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
