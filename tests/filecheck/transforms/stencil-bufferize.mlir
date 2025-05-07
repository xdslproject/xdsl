// RUN: xdsl-opt %s -p stencil-bufferize | filecheck %s

func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
  %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
    %5 = arith.constant 1.000000e+00 : f64
    %6 = arith.addf %4, %5 : f64
    stencil.return %6 : f64
  }
  stencil.store %3 to %2 (<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  func.return
}

// CHECK:         func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      stencil.apply(%3 = %0 : f64) outs (%2 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:        %4 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:        stencil.return %5 : f64
// CHECK-NEXT:      } to <[1, 2, 3], [65, 66, 63]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @bufferswapping(%f0 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %f1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.field<[-2,2002]x[-2,2002]xf32> {
  %time_m = arith.constant 0 : index
  %time_M = arith.constant 1001 : index
  %step = arith.constant 1 : index
  %t1_out, %t0_out = scf.for %time = %time_m to %time_M step %step iter_args(%fim1 = %f0, %fi = %f1) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) {
    %tim1 = stencil.load %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32> -> !stencil.temp<[0,2000]x[0,2000]xf32>
    %ti = stencil.apply(%tim1_b = %tim1 : !stencil.temp<[0,2000]x[0,2000]xf32>) -> (!stencil.temp<[0,2000]x[0,2000]xf32>) {
      %i = stencil.access %tim1_b[0, 0] : !stencil.temp<[0,2000]x[0,2000]xf32>
      stencil.return %i : f32
    }
    stencil.store %ti to %fi (<[0, 0], [2000, 2000]>) : !stencil.temp<[0,2000]x[0,2000]xf32> to !stencil.field<[-2,2002]x[-2,2002]xf32>
    scf.yield %fi, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>
  }
  func.return %t1_out : !stencil.field<[-2,2002]x[-2,2002]xf32>
}

// CHECK:         func.func @bufferswapping(%f0 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %f1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.field<[-2,2002]x[-2,2002]xf32> {
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1001 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %t1_out, %t0_out = scf.for %time = %time_m to %time_M step %step iter_args(%fim1 = %f0, %fi = %f1) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) {
// CHECK-NEXT:        stencil.apply(%tim1_b = %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) outs (%fi : !stencil.field<[-2,2002]x[-2,2002]xf32>) {
// CHECK-NEXT:          %i = stencil.access %tim1_b[0, 0] : !stencil.field<[-2,2002]x[-2,2002]xf32>
// CHECK-NEXT:          stencil.return %i : f32
// CHECK-NEXT:        } to <[0, 0], [2000, 2000]>
// CHECK-NEXT:        scf.yield %fi, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %t1_out : !stencil.field<[-2,2002]x[-2,2002]xf32>
// CHECK-NEXT:    }

func.func @copy_1d(%0 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
  %1 = stencil.cast %0 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
  %outc = stencil.cast %out : !stencil.field<?xf64> -> !stencil.field<[0,1024]xf64>
  %2 = stencil.load %1 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,68]xf64>
  %3 = stencil.apply(%4 = %2 : !stencil.temp<[-1,68]xf64>) -> (!stencil.temp<[0,68]xf64>) {
    %5 = stencil.access %4[-1] : !stencil.temp<[-1,68]xf64>
    stencil.return %5 : f64
  }
  stencil.store %3 to %outc (<[0], [68]>) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
  func.return
}

// CHECK:         func.func @copy_1d(%0 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
// CHECK-NEXT:      %1 = stencil.cast %0 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %outc = stencil.cast %out : !stencil.field<?xf64> -> !stencil.field<[0,1024]xf64>
// CHECK-NEXT:      stencil.apply(%2 = %1 : !stencil.field<[-4,68]xf64>) outs (%outc : !stencil.field<[0,1024]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %2[-1] : !stencil.field<[-4,68]xf64>
// CHECK-NEXT:        stencil.return %3 : f64
// CHECK-NEXT:      } to <[0], [68]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @copy_2d(%0 : !stencil.field<?x?xf64>, %out : !stencil.field<[-4,68]x[-4,68]xf64>) {
  %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
  %2 = stencil.load %1 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,64]x[0,68]xf64>
  %3 = stencil.apply(%4 = %2 : !stencil.temp<[-1,64]x[0,68]xf64>) -> (!stencil.temp<[0,64]x[0,68]xf64>) {
    %5 = stencil.access %4[-1, 0] : !stencil.temp<[-1,64]x[0,68]xf64>
    stencil.return %5 : f64
  }
  stencil.store %3 to %out (<[0, 0], [64, 68]>) : !stencil.temp<[0,64]x[0,68]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>
  func.return
}

// CHECK:         func.func @copy_2d(%0 : !stencil.field<?x?xf64>, %out : !stencil.field<[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:      %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      stencil.apply(%2 = %1 : !stencil.field<[-4,68]x[-4,68]xf64>) outs (%out : !stencil.field<[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %2[-1, 0] : !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        stencil.return %3 : f64
// CHECK-NEXT:      } to <[0, 0], [64, 68]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


func.func @copy_3d(%0 : !stencil.field<?x?x?xf64>, %out : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
  %1 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
  %2 = stencil.load %1 : !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64> -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
  %3 = stencil.apply(%4 = %2 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,68]xf64>) {
    %5 = stencil.access %4[-1, 0, 1] : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
    stencil.return %5 : f64
  }
  stencil.store %3 to %out (<[0, 0, 0], [64, 64, 68]>) : !stencil.temp<[0,64]x[0,64]x[0,68]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  func.return
}

// CHECK:         func.func @copy_3d(%0 : !stencil.field<?x?x?xf64>, %out : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:      %1 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
// CHECK-NEXT:      stencil.apply(%2 = %1 : !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>) outs (%out : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %2[-1, 0, 1] : !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
// CHECK-NEXT:        stencil.return %3 : f64
// CHECK-NEXT:      } to <[0, 0, 0], [64, 64, 68]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @offsets(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>, %2 : !stencil.field<?x?x?xf64>) {
  %3 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %4 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %5 = stencil.cast %2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %6 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
  %7, %8 = stencil.apply(%9 = %6 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
    %10 = stencil.access %9[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %11 = stencil.access %9[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %12 = stencil.access %9[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %13 = stencil.access %9[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %14 = stencil.access %9[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %15 = arith.addf %10, %11 : f64
    %16 = arith.addf %12, %13 : f64
    %17 = arith.addf %15, %16 : f64
    %cst = arith.constant -4.000000e+00 : f64
    %18 = arith.mulf %14, %cst : f64
    %19 = arith.addf %18, %17 : f64
    stencil.return %19, %18 : f64, f64
  }
  stencil.store %7 to %4 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  func.return
}

// CHECK:         func.func @offsets(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>, %2 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %3 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      stencil.apply(%5 = %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%4 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:        %6 = stencil.access %5[-1, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        %7 = stencil.access %5[1, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        %8 = stencil.access %5[0, 1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        %9 = stencil.access %5[0, -1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        %10 = stencil.access %5[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:        %11 = arith.addf %6, %7 : f64
// CHECK-NEXT:        %12 = arith.addf %8, %9 : f64
// CHECK-NEXT:        %13 = arith.addf %11, %12 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %14 = arith.mulf %10, %cst : f64
// CHECK-NEXT:        %15 = arith.addf %14, %13 : f64
// CHECK-NEXT:        stencil.return %15 : f64
// CHECK-NEXT:      } to <[0, 0, 0], [64, 64, 64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
  %2 = stencil.load %0 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
  %3 = stencil.apply(%4 = %2 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[1,65]xf64>) {
    %5 = stencil.access %4[-1] : !stencil.temp<[0,64]xf64>
    stencil.return %5 : f64
  }
  %4 = stencil.buffer %3 : !stencil.temp<[1,65]xf64> -> !stencil.temp<[1,65]xf64>
  %5 = stencil.apply(%6 = %4 : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
    %7 = stencil.access %6[1] : !stencil.temp<[1,65]xf64>
    stencil.return %7 : f64
  }
  stencil.store %5 to %1 (<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
  func.return
}

// CHECK:         func.func @stencil_buffer(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %2 = stencil.alloc : !stencil.field<[1,65]xf64>
// CHECK-NEXT:      stencil.apply(%3 = %0 : !stencil.field<[-4,68]xf64>) outs (%2 : !stencil.field<[1,65]xf64>) {
// CHECK-NEXT:        %4 = stencil.access %3[-1] : !stencil.field<[-4,68]xf64>
// CHECK-NEXT:        stencil.return %4 : f64
// CHECK-NEXT:      } to <[1], [65]>
// CHECK-NEXT:      stencil.apply(%3 = %2 : !stencil.field<[1,65]xf64>) outs (%1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:        %4 = stencil.access %3[1] : !stencil.field<[1,65]xf64>
// CHECK-NEXT:        stencil.return %4 : f64
// CHECK-NEXT:      } to <[0], [64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @stencil_two_stores(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>, %2 : !stencil.field<[-4,68]xf64>) {
  %3 = stencil.load %0 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
  %4 = stencil.apply(%5 = %3 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[1,65]xf64>) {
    %6 = stencil.access %5[-1] : !stencil.temp<[0,64]xf64>
    stencil.return %6 : f64
  }
  stencil.store %4 to %2 (<[1], [65]>) : !stencil.temp<[1,65]xf64> to !stencil.field<[-4,68]xf64>
  %5 = stencil.apply(%6 = %4 : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
    %7 = stencil.access %6[1] : !stencil.temp<[1,65]xf64>
    stencil.return %7 : f64
  }
  stencil.store %5 to %1 (<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
  func.return
}

// CHECK:         func.func @stencil_two_stores(%0 : !stencil.field<[-4,68]xf64>, %1 : !stencil.field<[-4,68]xf64>, %2 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      stencil.apply(%3 = %0 : !stencil.field<[-4,68]xf64>) outs (%2 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:        %4 = stencil.access %3[-1] : !stencil.field<[-4,68]xf64>
// CHECK-NEXT:        stencil.return %4 : f64
// CHECK-NEXT:      } to <[1], [65]>
// CHECK-NEXT:      stencil.apply(%3 = %2 : !stencil.field<[-4,68]xf64>) outs (%1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:        %4 = stencil.access %3[1] : !stencil.field<[-4,68]xf64>
// CHECK-NEXT:        stencil.return %4 : f64
// CHECK-NEXT:      } to <[0], [64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @apply_kernel(%0 : !stencil.field<[-2,13]x[-2,13]xf32>, %1 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_1", "u_vec", "timers"]} {
  %2 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
  %u_vec = builtin.unrealized_conversion_cast %2 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
  %3 = builtin.unrealized_conversion_cast %1 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
  "gpu.memcpy"(%2, %3) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
  %4 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
  %u_vec_1 = builtin.unrealized_conversion_cast %4 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
  %5 = builtin.unrealized_conversion_cast %0 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
  "gpu.memcpy"(%4, %5) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
  %time_m = arith.constant 0 : index
  %time_M = arith.constant 10 : index
  %step = arith.constant 1 : index
  %6, %7 = scf.for %time = %time_m to %time_M step %step iter_args(%t0 = %u_vec_1, %t1 = %u_vec) -> (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) {
    %t0_temp = stencil.load %t0 : !stencil.field<[-2,13]x[-2,13]xf32> -> !stencil.temp<[0,11]x[0,11]xf32>
    %t1_result = stencil.apply(%t0_buff = %t0_temp : !stencil.temp<[0,11]x[0,11]xf32>) -> (!stencil.temp<[0,11]x[0,11]xf32>) {
      %8 = stencil.access %t0_buff[0, 0] : !stencil.temp<[0,11]x[0,11]xf32>
      stencil.return %8 : f32
    }
    stencil.store %t1_result to %t1 (<[0, 0], [11, 11]>) : !stencil.temp<[0,11]x[0,11]xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
    scf.yield %t1, %t0 : !stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>
  }
  func.return
}

// CHECK:         func.func @apply_kernel(%0 : !stencil.field<[-2,13]x[-2,13]xf32>, %1 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr)  attributes {param_names = ["u_vec_1", "u_vec", "timers"]} {
// CHECK-NEXT:      %2 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec = builtin.unrealized_conversion_cast %2 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
// CHECK-NEXT:      %3 = builtin.unrealized_conversion_cast %1 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%2, %3) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %4 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec_1 = builtin.unrealized_conversion_cast %4 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
// CHECK-NEXT:      %5 = builtin.unrealized_conversion_cast %0 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%4, %5) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 10 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %6, %7 = scf.for %time = %time_m to %time_M step %step iter_args(%t0 = %u_vec_1, %t1 = %u_vec) -> (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) {
// CHECK-NEXT:        stencil.apply(%t0_buff = %t0 : !stencil.field<[-2,13]x[-2,13]xf32>) outs (%t1 : !stencil.field<[-2,13]x[-2,13]xf32>) {
// CHECK-NEXT:          %8 = stencil.access %t0_buff[0, 0] : !stencil.field<[-2,13]x[-2,13]xf32>
// CHECK-NEXT:          stencil.return %8 : f32
// CHECK-NEXT:        } to <[0, 0], [11, 11]>
// CHECK-NEXT:        scf.yield %t1, %t0 : !stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @stencil_init_float_unrolled(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
  %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
    %5 = arith.constant 1.000000e+00 : f64
    %6 = arith.constant 2.000000e+00 : f64
    %7 = arith.constant 3.000000e+00 : f64
    %8 = arith.constant 4.000000e+00 : f64
    %9 = arith.constant 5.000000e+00 : f64
    %10 = arith.constant 6.000000e+00 : f64
    %11 = arith.constant 7.000000e+00 : f64
    %12 = arith.constant 8.000000e+00 : f64
    stencil.return %5, %6, %7, %8, %9, %10, %11, %12 unroll <[2, 2, 2]> : f64, f64, f64, f64, f64, f64, f64, f64
  }
  stencil.store %3 to %2 (<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  func.return
}

// CHECK:         func.func @stencil_init_float_unrolled(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      stencil.apply(%3 = %0 : f64) outs (%2 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:        %4 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %5 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        %6 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:        %7 = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:        %8 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:        %9 = arith.constant 6.000000e+00 : f64
// CHECK-NEXT:        %10 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:        %11 = arith.constant 8.000000e+00 : f64
// CHECK-NEXT:        stencil.return %4, %5, %6, %7, %8, %9, %10, %11 unroll <[2, 2, 2]> : f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      } to <[1, 2, 3], [65, 66, 63]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @stencil_init_index(%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
  %1 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
    %x = stencil.index 0 <[0, 0, 0]>
    %y = stencil.index 1 <[0, 0, 0]>
    %z = stencil.index 2 <[0, 0, 0]>
    %xy = arith.addi %x, %y : index
    %xyz = arith.addi %xy, %z : index
    stencil.return %xyz : index
  }
  stencil.store %1 to %0 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
  func.return
}

// CHECK:         func.func @stencil_init_index(%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
// CHECK-NEXT:      stencil.apply() outs (%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
// CHECK-NEXT:        %x = stencil.index 0 <[0, 0, 0]>
// CHECK-NEXT:        %y = stencil.index 1 <[0, 0, 0]>
// CHECK-NEXT:        %z = stencil.index 2 <[0, 0, 0]>
// CHECK-NEXT:        %xy = arith.addi %x, %y : index
// CHECK-NEXT:        %xyz = arith.addi %xy, %z : index
// CHECK-NEXT:        stencil.return %xyz : index
// CHECK-NEXT:      } to <[0, 0, 0], [64, 64, 64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @stencil_init_index_offset(%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
  %1 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
    %x = stencil.index 0 <[1, 1, 1]>
    %y = stencil.index 1 <[-1, -1, -1]>
    %z = stencil.index 2 <[0, 0, 0]>
    %xy = arith.addi %x, %y : index
    %xyz = arith.addi %xy, %z : index
    stencil.return %xyz : index
  }
  stencil.store %1 to %0 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
  func.return
}

// CHECK:         func.func @stencil_init_index_offset(%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
// CHECK-NEXT:      stencil.apply() outs (%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
// CHECK-NEXT:        %x = stencil.index 0 <[1, 1, 1]>
// CHECK-NEXT:        %y = stencil.index 1 <[-1, -1, -1]>
// CHECK-NEXT:        %z = stencil.index 2 <[0, 0, 0]>
// CHECK-NEXT:        %xy = arith.addi %x, %y : index
// CHECK-NEXT:        %xyz = arith.addi %xy, %z : index
// CHECK-NEXT:        stencil.return %xyz : index
// CHECK-NEXT:      } to <[0, 0, 0], [64, 64, 64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @if_lowering(%arg0 : f64, %b0 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>, %b1 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>)  attributes {"stencil.program"} {
  %0, %1 = stencil.apply(%arg1 = %arg0 : f64) -> (!stencil.temp<[0,7]x[0,7]x[0,7]xf64>, !stencil.temp<[0,7]x[0,7]x[0,7]xf64>) {
    %true = "test.pureop"() : () -> i1
    %2, %3 = scf.if %true -> (!stencil.result<f64>, f64) {
      %4 = stencil.store_result %arg1 : !stencil.result<f64>
      scf.yield %4, %arg1 : !stencil.result<f64>, f64
    } else {
      %5 = stencil.store_result  : !stencil.result<f64>
      scf.yield %5, %arg1 : !stencil.result<f64>, f64
    }
    %6 = stencil.store_result %3 : !stencil.result<f64>
    stencil.return %2, %6 : !stencil.result<f64>, !stencil.result<f64>
  }
  stencil.store %0 to %b0 (<[0, 0, 0], [7, 7, 7]>) : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> to !stencil.field<[0,7]x[0,7]x[0,7]xf64>
  stencil.store %1 to %b1 (<[0, 0, 0], [7, 7, 7]>) : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> to !stencil.field<[0,7]x[0,7]x[0,7]xf64>
  func.return
}

// CHECK:         func.func @if_lowering(%arg0 : f64, %b0 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>, %b1 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      stencil.apply(%arg1 = %arg0 : f64) outs (%b0 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>, %b1 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>) {
// CHECK-NEXT:        %true = "test.pureop"() : () -> i1
// CHECK-NEXT:        %0, %1 = scf.if %true -> (!stencil.result<f64>, f64) {
// CHECK-NEXT:          %2 = stencil.store_result %arg1 : !stencil.result<f64>
// CHECK-NEXT:          scf.yield %2, %arg1 : !stencil.result<f64>, f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %3 = stencil.store_result  : !stencil.result<f64>
// CHECK-NEXT:          scf.yield %3, %arg1 : !stencil.result<f64>, f64
// CHECK-NEXT:        }
// CHECK-NEXT:        %4 = stencil.store_result %1 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %0, %4 : !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      } to <[0, 0, 0], [7, 7, 7]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @combine(%0 : !stencil.field<?x?xf64>) {
  %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
  %2 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
    %3 = arith.constant 1.000000e+00 : f64
    stencil.return %3 : f64
  }
  %3 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
    %4 = arith.constant 2.000000e+00 : f64
    stencil.return %4 : f64
  }
  %4 = stencil.combine 0 at 33 lower = (%2 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%3 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
  stencil.store %4 to %1 (<[1, 2], [65, 66]>) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
  func.return
}

// CHECK:         func.func @combine(%0 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      stencil.apply() outs (%1 : !stencil.field<[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:        %2 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        stencil.return %2 : f64
// CHECK-NEXT:      } to <[1, 2], [33, 66]>
// CHECK-NEXT:      stencil.apply() outs (%1 : !stencil.field<[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:        %2 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        stencil.return %2 : f64
// CHECK-NEXT:      } to <[33, 2], [65, 66]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @buffered_combine(%0 : !stencil.field<?x?xf64>) {
  %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
  %2 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
    %3 = arith.constant 1.000000e+00 : f64
    stencil.return %3 : f64
  }
  %3 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
    %4 = arith.constant 2.000000e+00 : f64
    stencil.return %4 : f64
  }
  %4 = stencil.combine 0 at 33 lower = (%2 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%3 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
  %5 = stencil.buffer %4 : !stencil.temp<[1,65]x[2,66]xf64> -> !stencil.temp<[1,65]x[2,66]xf64>
  %6 = stencil.apply(%7 = %5 : !stencil.temp<[1,65]x[2,66]xf64>) -> (!stencil.temp<[1,65]x[2,66]xf64>) {
    %8 = arith.constant 1.000000e+00 : f64
    %9 = stencil.access %7[0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
    %10 = arith.addf %8, %9 : f64
    stencil.return %10 : f64
  }
  stencil.store %6 to %1 (<[1, 2], [65, 66]>) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
  func.return
}

// CHECK:         func.func @buffered_combine(%0 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %1 = stencil.alloc : !stencil.field<[1,65]x[2,66]xf64>
// CHECK-NEXT:      %2 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      stencil.apply() outs (%1 : !stencil.field<[1,65]x[2,66]xf64>) {
// CHECK-NEXT:        %3 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        stencil.return %3 : f64
// CHECK-NEXT:      } to <[1, 2], [33, 66]>
// CHECK-NEXT:      stencil.apply() outs (%1 : !stencil.field<[1,65]x[2,66]xf64>) {
// CHECK-NEXT:        %3 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        stencil.return %3 : f64
// CHECK-NEXT:      } to <[33, 2], [65, 66]>
// CHECK-NEXT:      stencil.apply(%3 = %1 : !stencil.field<[1,65]x[2,66]xf64>) outs (%2 : !stencil.field<[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:        %4 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %5 = stencil.access %3[0, 0] : !stencil.field<[1,65]x[2,66]xf64>
// CHECK-NEXT:        %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:        stencil.return %6 : f64
// CHECK-NEXT:      } to <[1, 2], [65, 66]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// This should bufferize as an in-place increment
// The accesses are zero-offset, so it's safe to do in-place
func.func @increment_n(%0 : !stencil.field<[0,64]x[0,64]xf64>, %n : f64) {
  %load = stencil.load %0 : !stencil.field<[0,64]x[0,64]xf64> -> !stencil.temp<[0,64]x[0,64]xf64>
  %inplace = stencil.apply(%load_arg = %load : !stencil.temp<[0,64]x[0,64]xf64>, %nn = %n : f64) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
    %acc = stencil.access %load_arg[0, 0] : !stencil.temp<[0,64]x[0,64]xf64>
    %inc = arith.addf %acc, %nn : f64
    stencil.return %inc : f64
  }
  stencil.store %inplace to %0(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[0,64]x[0,64]xf64>
  func.return
}

// CHECK:    func.func @increment_n(%inplace : !stencil.field<[0,64]x[0,64]xf64>, %n : f64) {
// CHECK-NEXT:      stencil.apply(%load_arg = %inplace : !stencil.field<[0,64]x[0,64]xf64>, %nn = %n : f64) outs (%inplace : !stencil.field<[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %acc = stencil.access %load_arg[0, 0] : !stencil.field<[0,64]x[0,64]xf64>
// CHECK-NEXT:        %inc = arith.addf %acc, %nn : f64
// CHECK-NEXT:        stencil.return %inc : f64
// CHECK-NEXT:      } to <[0, 0], [64, 64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// This should *not* bufferize as an in-place increment
// There is a non-zero offset on the access, so it wouldn't be safe to do in-place in parallel.
func.func @increment_n_offset(%0 : !stencil.field<[0,64]x[0,64]xf64>, %n : f64) {
  %load = stencil.load %0 : !stencil.field<[0,64]x[0,64]xf64> -> !stencil.temp<[0,64]x[0,64]xf64>
  %inplace = stencil.apply(%load_arg = %load : !stencil.temp<[0,64]x[0,64]xf64>, %nn = %n : f64) -> (!stencil.temp<[0,64]x[0,64]xf64>) {
    %acc = stencil.access %load_arg[0, 1] : !stencil.temp<[0,64]x[0,64]xf64>
    %inc = arith.addf %acc, %nn : f64
    stencil.return %inc : f64
  }
  stencil.store %inplace to %0(<[0, 0], [64, 64]>) : !stencil.temp<[0,64]x[0,64]xf64> to !stencil.field<[0,64]x[0,64]xf64>
  func.return
}

// CHECK:         func.func @increment_n_offset(%0 : !stencil.field<[0,64]x[0,64]xf64>, %n : f64) {
// CHECK-NEXT:      %load = stencil.load %0 : !stencil.field<[0,64]x[0,64]xf64> -> !stencil.temp<[0,64]x[0,64]xf64>
// CHECK-NEXT:      %inplace = stencil.buffer %load : !stencil.temp<[0,64]x[0,64]xf64> -> !stencil.field<[0,64]x[0,64]xf64>
// CHECK-NEXT:      stencil.apply(%load_arg = %inplace : !stencil.field<[0,64]x[0,64]xf64>, %nn = %n : f64) outs (%0 : !stencil.field<[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %acc = stencil.access %load_arg[0, 1] : !stencil.field<[0,64]x[0,64]xf64>
// CHECK-NEXT:        %inc = arith.addf %acc, %nn : f64
// CHECK-NEXT:        stencil.return %inc : f64
// CHECK-NEXT:      } to <[0, 0], [64, 64]>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
  %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
  %1 = "dmp.swap"(%0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
  %2 = stencil.apply(%3 = %1 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
    %4 = arith.constant dense<1.666600e-01> : tensor<510xf32>
    %5 = stencil.access %3[1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %6 = "tensor.extract_slice"(%5) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %7 = arith.addf %4, %6 : tensor<510xf32>
    stencil.return %7 : tensor<510xf32>
  }
  stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
  func.return
}

// CHECK:      func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:   "dmp.swap"(%a) {strategy = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, swaps = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-NEXT:   stencil.apply(%0 = %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) outs (%b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %1 = arith.constant dense<1.666600e-01> : tensor<510xf32>
// CHECK-NEXT:     %2 = stencil.access %0[1, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %3 = "tensor.extract_slice"(%2) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:     %4 = arith.addf %1, %3 : tensor<510xf32>
// CHECK-NEXT:     stencil.return %4 : tensor<510xf32>
// CHECK-NEXT:   } to <[0, 0], [1, 1]>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
