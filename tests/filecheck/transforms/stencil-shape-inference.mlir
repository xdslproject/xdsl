// RUN: xdsl-opt -p shape-inference --verify-diagnostics --split-input-file %s | filecheck %s

builtin.module {
  func.func @different_input_offsets(%out : !stencil.field<[-4,68]xf64>, %left : !stencil.field<[-4,68]xf64>, %right : !stencil.field<[-4,68]xf64>) {
    %tleft = stencil.load %left : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %tright = stencil.load %right : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %tout = stencil.apply(%lefti = %tleft : !stencil.temp<?xf64>, %righti = %tright : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %l = stencil.access %lefti[-1] : !stencil.temp<?xf64>
      %r = stencil.access %righti[1] : !stencil.temp<?xf64>
      %o = arith.addf %l, %r : f64
      stencil.return %o : f64
    }
    stencil.store %tout to %out(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }
}

// CHECK:         func.func @different_input_offsets(%out : !stencil.field<[-4,68]xf64>, %left : !stencil.field<[-4,68]xf64>, %right : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %tleft = stencil.load %left : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:      %tright = stencil.load %right : !stencil.field<[-4,68]xf64> -> !stencil.temp<[1,65]xf64>
// CHECK-NEXT:      %tout = stencil.apply(%lefti = %tleft : !stencil.temp<[-1,63]xf64>, %righti = %tright : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
// CHECK-NEXT:        %l = stencil.access %lefti[-1] : !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:        %r = stencil.access %righti[1] : !stencil.temp<[1,65]xf64>
// CHECK-NEXT:        %o = arith.addf %l, %r : f64
// CHECK-NEXT:        stencil.return %o : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %tout to %out(<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
      %7 = stencil.access %6[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %8 = stencil.access %6[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %9 = stencil.access %6[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %10 = stencil.access %6[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %11 = stencil.access %6[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      stencil.return %16 : f64
    }
    stencil.store %5 to %3(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }
}


// CHECK:         func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %3 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %7 = stencil.access %6[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %8 = stencil.access %6[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %9 = stencil.access %6[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %10 = stencil.access %6[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %11 = stencil.access %6[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %12 = arith.addf %7, %8 : f64
// CHECK-NEXT:        %13 = arith.addf %9, %10 : f64
// CHECK-NEXT:        %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %15 = arith.mulf %11, %cst : f64
// CHECK-NEXT:        %16 = arith.addf %15, %14 : f64
// CHECK-NEXT:        stencil.return %16 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %3(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func @stencil_hdiff(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[0,68]x[0,68]x[0,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[0,68]x[0,68]x[0,68]xf64> -> !stencil.temp<?x?x?xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %7 = stencil.access %6[-1, 0, 0] : !stencil.temp<?x?x?xf64>
      %8 = stencil.access %6[1, 0, 0] : !stencil.temp<?x?x?xf64>
      %9 = stencil.access %6[0, 1, 0] : !stencil.temp<?x?x?xf64>
      %10 = stencil.access %6[0, -1, 0] : !stencil.temp<?x?x?xf64>
      %11 = stencil.access %6[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %15 = arith.mulf %11, %cst : f64
      %16 = arith.addf %15, %14 : f64
      stencil.return %16 : f64
    }
    stencil.store %5 to %3(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }
}

// CHECK:               %4 = "stencil.load"(%2) : (!stencil.field<[0,68]x[0,68]x[0,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:          ^^^^^^^^^^^^^^^^^^^-----------------------------------------------------------
// CHECK-NEXT:          | Operation does not verify: The stencil.load is too big for the loaded field.
// CHECK-NEXT:          ------------------------------------------------------------------------------
// -----

builtin.module {

  func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
      %5 = arith.constant 1.000000e+00 : f64
      %6 = arith.addf %4, %5 : f64
      stencil.return %6 : f64
    }
    stencil.store %3 to %2(<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    func.return
  }
}

// CHECK:         func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
// CHECK-NEXT:        %5 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:        stencil.return %6 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %2(<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func private @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
    %2 = stencil.load %0 : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %3 = stencil.apply(%4 = %2 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %5 = stencil.access %4[0] : !stencil.temp<?xf64>
      stencil.return %5 : f64
    }
    %6 = stencil.buffer %3 : !stencil.temp<?xf64> -> !stencil.temp<?xf64>
    %7 = stencil.apply(%8 = %6 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %9 = stencil.access %8[0] : !stencil.temp<?xf64>
      stencil.return %9 : f64
    }
    stencil.store %7 to %1(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }
}

// CHECK:         func.func private @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %2 = stencil.load %0 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%4 = %2 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
// CHECK-NEXT:        %5 = stencil.access %4[0] : !stencil.temp<[0,64]xf64>
// CHECK-NEXT:        stencil.return %5 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = stencil.buffer %3 : !stencil.temp<[0,64]xf64> -> !stencil.temp<[0,64]xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
// CHECK-NEXT:        %7 = stencil.access %6[0] : !stencil.temp<[0,64]xf64>
// CHECK-NEXT:        stencil.return %7 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %1(<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {"stencil.program"} {
    %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<?x?x?xf64>
    %3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %c0 = arith.constant 0 : index
      %4 = stencil.dyn_access %arg2[%c0, %c0, %c0] in <[-1, -2, 0]> : <[1, 2, 0]> : !stencil.temp<?x?x?xf64>
      %5 = stencil.store_result %4 : !stencil.result<f64>
      stencil.return %5 : !stencil.result<f64>
    }
    stencil.store %3 to %1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    func.return
  }

// CHECK:         func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%arg2 = %2 : !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %c0 = arith.constant 0 : index
// CHECK-NEXT:        %4 = stencil.dyn_access %arg2[%c0, %c0, %c0] in <[-1, -2, 0]> : <[1, 2, 0]> : !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>
// CHECK-NEXT:        %5 = stencil.store_result %4 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %5 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @combine(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {"stencil.program"} {
    %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<?x?x?xf64>
    %3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %4 = stencil.access %arg2[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %5 = stencil.store_result %4 : !stencil.result<f64>
      stencil.return %5 : !stencil.result<f64>
    }
    %6 = stencil.apply(%arg2_1 = %2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %7 = stencil.access %arg2_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %8 = stencil.store_result %7 : !stencil.result<f64>
      stencil.return %8 : !stencil.result<f64>
    }
    %9 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%6 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
    stencil.store %9 to %1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    func.return
  }

// CHECK:         func.func @combine(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%arg2 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,32]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %4 = stencil.access %arg2[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %5 = stencil.store_result %4 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %5 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = stencil.apply(%arg2 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[32,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %5 = stencil.access %arg2[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %6 = stencil.store_result %5 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %6 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<[0,32]x[0,64]x[0,60]xf64>) upper = (%4 : !stencil.temp<[32,64]x[0,64]x[0,60]xf64>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      stencil.store %5 to %1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @buffer(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>) attributes {"stencil.program"} {
    %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %3 = stencil.apply() -> (!stencil.temp<?x?x?xf64>) {
      %cst = arith.constant 1.000000e+00 : f64
      %4 = stencil.store_result %cst : !stencil.result<f64>
      stencil.return %4 : !stencil.result<f64>
    }
    %5 = stencil.apply() -> (!stencil.temp<?x?x?xf64>) {
      %cst_1 = arith.constant 1.000000e+00 : f64
      %6 = stencil.store_result %cst_1 : !stencil.result<f64>
      stencil.return %6 : !stencil.result<f64>
    }
    %7 = stencil.apply() -> (!stencil.temp<?x?x?xf64>) {
      %cst_2 = arith.constant 1.000000e+00 : f64
      %8 = stencil.store_result %cst_2 : !stencil.result<f64>
      stencil.return %8 : !stencil.result<f64>
    }
    %9, %10, %11 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%3 : !stencil.temp<?x?x?xf64>) lowerext = %7 : !stencil.temp<?x?x?xf64> upperext = %5 : !stencil.temp<?x?x?xf64> : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
    %12 = stencil.buffer %9 : !stencil.temp<?x?x?xf64> -> !stencil.temp<?x?x?xf64>
    %13 = stencil.buffer %10 : !stencil.temp<?x?x?xf64> -> !stencil.temp<?x?x?xf64>
    %14 = stencil.buffer %11 : !stencil.temp<?x?x?xf64> -> !stencil.temp<?x?x?xf64>
    %15 = stencil.apply(%arg3 = %12 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %16 = stencil.access %arg3[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %17 = stencil.store_result %16 : !stencil.result<f64>
      stencil.return %17 : !stencil.result<f64>
    }
    %18 = stencil.apply(%arg3_1 = %13 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %19 = stencil.access %arg3_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %20 = stencil.store_result %19 : !stencil.result<f64>
      stencil.return %20 : !stencil.result<f64>
    }
    %21 = stencil.apply(%arg3_2 = %14 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
      %22 = stencil.access %arg3_2[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %23 = stencil.store_result %22 : !stencil.result<f64>
      stencil.return %23 : !stencil.result<f64>
    }
    stencil.store %15 to %0(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    stencil.store %18 to %1(<[0, 0, 0], [16, 64, 60]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    stencil.store %21 to %2(<[48, 0, 0], [64, 64, 60]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    func.return
  }

// CHECK:         func.func @buffer(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %3 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %4 = stencil.store_result %cst : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %4 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = stencil.apply() -> (!stencil.temp<[32,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %5 = stencil.store_result %cst : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %5 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = stencil.apply() -> (!stencil.temp<[0,32]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %6 = stencil.store_result %cst : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %6 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %6, %7, %8 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) upper = (%3 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) lowerext = %5 : !stencil.temp<[0,32]x[0,64]x[0,60]xf64> upperext = %4 : !stencil.temp<[32,64]x[0,64]x[0,60]xf64> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.temp<[0,16]x[0,64]x[0,60]xf64>, !stencil.temp<[48,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %9 = stencil.buffer %6 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %10 = stencil.buffer %7 : !stencil.temp<[0,16]x[0,64]x[0,60]xf64> -> !stencil.temp<[0,16]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %11 = stencil.buffer %8 : !stencil.temp<[48,64]x[0,64]x[0,60]xf64> -> !stencil.temp<[48,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %12 = stencil.apply(%arg3 = %9 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %13 = stencil.access %arg3[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %14 = stencil.store_result %13 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %14 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %13 = stencil.apply(%arg3 = %10 : !stencil.temp<[0,16]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,16]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %14 = stencil.access %arg3[0, 0, 0] : !stencil.temp<[0,16]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %15 = stencil.store_result %14 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %15 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stencil.apply(%arg3 = %11 : !stencil.temp<[48,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[48,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %15 = stencil.access %arg3[0, 0, 0] : !stencil.temp<[48,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %16 = stencil.store_result %15 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %16 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %12 to %0(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      stencil.store %13 to %1(<[0, 0, 0], [16, 64, 60]>) : !stencil.temp<[0,16]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      stencil.store %14 to %2(<[48, 0, 0], [64, 64, 60]>) : !stencil.temp<[48,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// -----

func.func @stencil_missing_dims(%in : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %ybuf : !stencil.field<[-4,68]xf64>, %zbuf : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
  %intemp = stencil.load %in : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
  %0 = "dmp.swap"(%intemp) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = []} : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %ybuf_t = stencil.load %ybuf : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
  %1 = "dmp.swap"(%ybuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = []} : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
  %zbuf_t = stencil.load %zbuf : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
  %2 = "dmp.swap"(%zbuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = []} : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
  %res = stencil.apply(%inarg = %0 : !stencil.temp<?x?x?xf64>, %ybuf_arg = %1 : !stencil.temp<?xf64>, %zbuf_arg = %2 : !stencil.temp<?xf64>) -> (!stencil.temp<?x?x?xf64>) {
    %3 = stencil.access %ybuf_arg[_, -1, _] : !stencil.temp<?xf64>
    %4 = stencil.access %zbuf_arg[_, _, -1] : !stencil.temp<?xf64>
    %5 = stencil.access %inarg[0, -1, -1] : !stencil.temp<?x?x?xf64>
    %6 = arith.addf %3, %4 : f64
    %7 = arith.addf %5, %6 : f64
    stencil.return %7 : f64
  }
  stencil.store %res to %out(<[0, 0, 0], [1, 1, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  func.return
}

// CHECK:       func.func @stencil_missing_dims(%in : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %ybuf : !stencil.field<[-4,68]xf64>, %zbuf : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:    %intemp = stencil.load %in : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
// CHECK-NEXT:    %0 = "dmp.swap"(%intemp) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>) -> !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
// CHECK-NEXT:    %ybuf_t = stencil.load %ybuf : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,0]xf64>
// CHECK-NEXT:    %1 = "dmp.swap"(%ybuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,0]xf64>) -> !stencil.temp<[-1,0]xf64>
// CHECK-NEXT:    %zbuf_t = stencil.load %zbuf : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:    %2 = "dmp.swap"(%zbuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = []} : (!stencil.temp<[-1,63]xf64>) -> !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:    %res = stencil.apply(%inarg = %0 : !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>, %ybuf_arg = %1 : !stencil.temp<[-1,0]xf64>, %zbuf_arg = %2 : !stencil.temp<[-1,63]xf64>) -> (!stencil.temp<[0,1]x[0,1]x[0,64]xf64>) {
// CHECK-NEXT:      %3 = stencil.access %ybuf_arg[_, -1, _] : !stencil.temp<[-1,0]xf64>
// CHECK-NEXT:      %4 = stencil.access %zbuf_arg[_, _, -1] : !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:      %5 = stencil.access %inarg[0, -1, -1] : !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
// CHECK-NEXT:      %6 = arith.addf %3, %4 : f64
// CHECK-NEXT:      %7 = arith.addf %5, %6 : f64
// CHECK-NEXT:      stencil.return %7 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    stencil.store %res to %out(<[0, 0, 0], [1, 1, 64]>) : !stencil.temp<[0,1]x[0,1]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
