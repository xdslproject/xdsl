// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @stencil_copy(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
      %7 = stencil.access %6[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
      %8 = stencil.store_result %7 : !stencil.result<f64>
      stencil.return %8 : !stencil.result<f64>
    }
    stencil.store %5 to %3(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @stencil_copy(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %3 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %7 = stencil.access %6[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:        %8 = stencil.store_result %7 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %8 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %3(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func private @myfunc(%b0 : !stencil.field<?x?x?xf32>, %b1 : !stencil.field<?x?x?xf32>)  attributes {"param_names" = ["data"]} {
    %f0 = stencil.cast %b0 : !stencil.field<?x?x?xf32> -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
    %f1 = stencil.cast %b1 : !stencil.field<?x?x?xf32> -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
    %time_m = arith.constant 0 : index
    %time_M = arith.constant 1000 : index
    %step = arith.constant 1 : index
    %fnp1, %fn = scf.for %time = %time_m to %time_M step %step iter_args(%fi = %f0, %fip1 = %f1) -> (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) {
      %ti = stencil.load %fi : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32> -> !stencil.temp<?x?x?xf32>
      %tip1 = stencil.apply(%ti_ = %ti : !stencil.temp<?x?x?xf32>) -> (!stencil.temp<?x?x?xf32>) {
        %v = stencil.access %ti_[0, 0, 0] : !stencil.temp<?x?x?xf32>
        stencil.return %v : f32
      }
      stencil.store %tip1 to %fip1(<[0, 0, 0], [50, 80, 40]>) : !stencil.temp<?x?x?xf32> to !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
      scf.yield %fip1, %fi : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
    }
    func.return
  }
}

// CHECK:         func.func private @myfunc(%b0 : !stencil.field<?x?x?xf32>, %b1 : !stencil.field<?x?x?xf32>)  attributes {param_names = ["data"]} {
// CHECK-NEXT:      %f0 = stencil.cast %b0 : !stencil.field<?x?x?xf32> -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:      %f1 = stencil.cast %b1 : !stencil.field<?x?x?xf32> -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1000 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %fnp1, %fn = scf.for %time = %time_m to %time_M step %step iter_args(%fi = %f0, %fip1 = %f1) -> (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) {
// CHECK-NEXT:        %ti = stencil.load %fi : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32> -> !stencil.temp<?x?x?xf32>
// CHECK-NEXT:        %tip1 = stencil.apply(%ti_ = %ti : !stencil.temp<?x?x?xf32>) -> (!stencil.temp<?x?x?xf32>) {
// CHECK-NEXT:          %v = stencil.access %ti_[0, 0, 0] : !stencil.temp<?x?x?xf32>
// CHECK-NEXT:          stencil.return %v : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        stencil.store %tip1 to %fip1(<[0, 0, 0], [50, 80, 40]>) : !stencil.temp<?x?x?xf32> to !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:        scf.yield %fip1, %fi : !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func private @stencil_laplace(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]x[-1,65]xf64>) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
      %7 = stencil.access %6[-1, 0] : !stencil.temp<[-1,65]x[-1,65]xf64>
      %8 = stencil.access %6[1, 0] : !stencil.temp<[-1,65]x[-1,65]xf64>
      %9 = stencil.access %6[0, 1] : !stencil.temp<[-1,65]x[-1,65]xf64>
      %10 = stencil.access %6[0, -1] : !stencil.temp<[-1,65]x[-1,65]xf64>
      %11 = stencil.access %6[0, 0] : !stencil.temp<[-1,65]x[-1,65]xf64>
      %12 = arith.addf %7, %8 : f64
      %13 = arith.addf %9, %10 : f64
      %14 = arith.addf %12, %13 : f64
      %15 = arith.constant -4.000000e+00 : f64
      %16 = arith.mulf %11, %15 : f64
      %17 = arith.mulf %16, %13 : f64
      stencil.return %17 : f64
    }
    stencil.store %5 to %3(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func private @stencil_laplace(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %3 = stencil.cast %1 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]x[-1,65]xf64>) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %7 = stencil.access %6[-1, 0] : !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:        %8 = stencil.access %6[1, 0] : !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:        %9 = stencil.access %6[0, 1] : !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:        %10 = stencil.access %6[0, -1] : !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:        %11 = stencil.access %6[0, 0] : !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:        %12 = arith.addf %7, %8 : f64
// CHECK-NEXT:        %13 = arith.addf %9, %10 : f64
// CHECK-NEXT:        %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:        %15 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %16 = arith.mulf %11, %15 : f64
// CHECK-NEXT:        %17 = arith.mulf %16, %13 : f64
// CHECK-NEXT:        stencil.return %17 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %3(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
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
    %10 = stencil.combine 0 at 11 lower = (%6 : !stencil.temp<?xf64>) upper = (%7 : !stencil.temp<?xf64>) : !stencil.temp<?xf64>
    stencil.store %10 to %1(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func private @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %2 = stencil.load %0 : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %3 = stencil.apply(%4 = %2 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %5 = stencil.access %4[0] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %5 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = stencil.buffer %3 : !stencil.temp<?xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %7 = stencil.access %6[0] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %7 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = stencil.combine 0 at 11 lower = (%4 : !stencil.temp<?xf64>) upper = (%5 : !stencil.temp<?xf64>) : !stencil.temp<?xf64>
// CHECK-NEXT:      stencil.store %6 to %1(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func private @stencil_offset_mapping(%0 : !stencil.field<?xf64>, %1 : !stencil.field<?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,65]xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]xf64>) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
      %7 = stencil.access %6[_, 0] : !stencil.temp<[-1,65]xf64>
      %8 = stencil.access %6[0, _] : !stencil.temp<[-1,65]xf64>
      stencil.return %8 : f64
    }
    stencil.store %5 to %3(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func private @stencil_offset_mapping(%0 : !stencil.field<?xf64>, %1 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %0 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %3 = stencil.cast %1 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.load %2 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,65]xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]xf64>) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %7 = stencil.access %6[_, 0] : !stencil.temp<[-1,65]xf64>
// CHECK-NEXT:        %8 = stencil.access %6[0, _] : !stencil.temp<[-1,65]xf64>
// CHECK-NEXT:        stencil.return %8 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %3(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

builtin.module {
  func.func private @stencil_dyn_access(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>) {
    %2 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %3 = stencil.cast %1 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]xf64>
    %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]x[-1,65]xf64>) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
      %i = stencil.index 0 <[-1, -1]>
      %j = stencil.index 1 <[1, 1]>
      %7 = stencil.dyn_access %6[%i, %j] in <[-1, -1]> : <[1, 1]> : !stencil.temp<[-1,65]x[-1,65]xf64>
      stencil.return %7 : f64
    }
    stencil.store %5 to %3(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func private @stencil_dyn_access(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %3 = stencil.cast %1 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:      %5 = stencil.apply(%6 = %4 : !stencil.temp<[-1,65]x[-1,65]xf64>) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %i = stencil.index 0 <[-1, -1]>
// CHECK-NEXT:        %j = stencil.index 1 <[1, 1]>
// CHECK-NEXT:        %7 = stencil.dyn_access %6[%i, %j] in <[-1, -1]> : <[1, 1]> : !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:        stencil.return %7 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %3(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

builtin.module {
  func.func @stencil_copy_bufferized(%0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
    stencil.apply(%6 = %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
      %7 = stencil.access %6[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
      %8 = stencil.store_result %7 : !stencil.result<f64>
      stencil.return %8 : !stencil.result<f64>
    } to <[0, 0, 0], [64, 64, 64]>
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @stencil_copy_bufferized(%0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:      stencil.apply(%2 = %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %2[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        %4 = stencil.store_result %3 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %4 : !stencil.result<f64>
// CHECK-NEXT:      } to <[0, 0, 0], [64, 64, 64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
