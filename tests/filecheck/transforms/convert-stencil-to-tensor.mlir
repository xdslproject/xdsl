// RUN: xdsl-opt %s -p convert-stencil-to-tensor | filecheck %s

builtin.module {

  // The pass used to crash on external function, just regression-testing this here.
  func.func private @external(!stencil.field<?xf64>) -> ()

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

  // func.func @bufferswapping(%f0 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %f1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.field<[-2,2002]x[-2,2002]xf32> {
  //   %time_m = arith.constant 0 : index
  //   %time_M = arith.constant 1001 : index
  //   %step = arith.constant 1 : index
  //   %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
  //   ^1(%time : index, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %fi : !stencil.field<[-2,2002]x[-2,2002]xf32>):
  //     %tim1 = "stencil.load"(%fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.temp<[0,2000]x[0,2000]xf32>
  //     %ti = "stencil.apply"(%tim1) ({
  //     ^2(%tim1_b : !stencil.temp<[0,2000]x[0,2000]xf32>):
  //       %i = "stencil.access"(%tim1_b) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> f32
  //       "stencil.return"(%i) : (f32) -> ()
  //     }) : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> !stencil.temp<[0,2000]x[0,2000]xf32>
  //     "stencil.store"(%ti, %fi) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<2000, 2000>} : (!stencil.temp<[0,2000]x[0,2000]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
  //     "scf.yield"(%fi, %fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
  //   }) : (index, index, index, !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>)
  //   func.return %t1_out : !stencil.field<[-2,2002]x[-2,2002]xf32>
  // }

  // func.func @copy_1d(%0 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
  //   %1 = "stencil.cast"(%0) : (!stencil.field<?xf64>) -> !stencil.field<[-4,68]xf64>
  //   %outc = "stencil.cast"(%out) : (!stencil.field<?xf64>) -> !stencil.field<[0,1024]xf64>
  //   %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
  //   %3 = "stencil.apply"(%2) ({
  //   ^0(%4 : !stencil.temp<[-1,68]xf64>):
  //     %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1>} : (!stencil.temp<[-1,68]xf64>) -> f64
  //     "stencil.return"(%5) : (f64) -> ()
  //   }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
  //   "stencil.store"(%3, %outc) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<[0,68]xf64>, !stencil.field<[0,1024]xf64>) -> ()
  //   func.return
  // }

  // func.func @copy_2d(%0 : !stencil.field<?x?xf64>) {
  //   %1 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
  //   %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,64]x[0,68]xf64>
  //   %3 = "stencil.apply"(%2) ({
  //   ^0(%4 : !stencil.temp<[-1,64]x[0,68]xf64>):
  //     %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,64]x[0,68]xf64>) -> f64
  //     "stencil.return"(%5) : (f64) -> ()
  //   }) : (!stencil.temp<[-1,64]x[0,68]xf64>) -> !stencil.temp<[0,64]x[0,68]xf64>
  //   func.return
  // }


  // func.func @copy_3d(%0 : !stencil.field<?x?x?xf64>) {
  //   %1 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
  //   %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>) -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
  //   %3 = "stencil.apply"(%2) ({
  //   ^0(%4 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>):
  //     %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0, 1>} : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> f64
  //     "stencil.return"(%5) : (f64) -> ()
  //   }) : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,68]xf64>
  //   func.return
  // }

  // func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
  //   func.return
  // }
  // func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
  //   func.return
  // }

  func.func @offsets(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>, %2 : !stencil.field<?x?x?xf64>) {
    %3 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %5 = "stencil.cast"(%2) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %6 = "stencil.load"(%3) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %7, %8 = "stencil.apply"(%6) ({
    // %7 = "stencil.apply"(%6) ({
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
      // "stencil.return"(%19) : (f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>)
    // }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>)
    "stencil.store"(%7, %4) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "stencil.store"(%8, %5) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }

  // func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
  //     "stencil.external_store"(%dyn_field, %dyn_mem) : (!stencil.field<?x?x?xf64>, memref<?x?x?xf64>) -> ()
  //     "stencil.external_store"(%sta_field, %sta_mem) : (!stencil.field<[-2,62]x[0,64]x[2,66]xf64>, memref<64x64x64xf64>) -> ()
  //     %0 = "stencil.external_load"(%dyn_mem) : (memref<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
  //     %1 = "stencil.external_load"(%sta_mem) : (memref<64x64x64xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>

  //     %casted = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
  //     func.return
  // }

  // func.func @neg_bounds(%in : !stencil.field<[-32,32]xf64>, %out : !stencil.field<[-32,32]xf64>) {
  //   %tin = "stencil.load"(%in) : (!stencil.field<[-32,32]xf64>) -> !stencil.temp<[-16,16]xf64>
  //   %outt = "stencil.apply"(%tin) ({
  //   ^0(%tinb : !stencil.temp<[-16,16]xf64>):
  //     %val = "stencil.access"(%tinb) {"offset" = #stencil.index<0>} : (!stencil.temp<[-16,16]xf64>) -> f64
  //     "stencil.return"(%val) : (f64) -> ()
  //   }) : (!stencil.temp<[-16,16]xf64>) -> !stencil.temp<[-16,16]xf64>
  //   "stencil.store"(%outt, %out) {"lb" = #stencil.index<-16>, "ub" = #stencil.index<16>} : (!stencil.temp<[-16,16]xf64>, !stencil.field<[-32,32]xf64>) -> ()
  //   func.return
  // }

  // func.func @stencil_buffer(%49 : !stencil.field<[-4,68]xf64>, %50 : !stencil.field<[-4,68]xf64>) {
  //   %51 = "stencil.load"(%49) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[0,64]xf64>
  //   %52 = "stencil.apply"(%51) ({
  //   ^8(%53 : !stencil.temp<[0,64]xf64>):
  //     %54 = "stencil.access"(%53) {"offset" = #stencil.index<-1>} : (!stencil.temp<[0,64]xf64>) -> f64
  //     "stencil.return"(%54) : (f64) -> ()
  //   }) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[1,65]xf64>
  //   %55 = "stencil.buffer"(%52) : (!stencil.temp<[1,65]xf64>) -> !stencil.temp<[1,65]xf64>
  //   %56 = "stencil.apply"(%55) ({
  //   ^9(%57 : !stencil.temp<[1,65]xf64>):
  //     %58 = "stencil.access"(%57) {"offset" = #stencil.index<1>} : (!stencil.temp<[1,65]xf64>) -> f64
  //     "stencil.return"(%58) : (f64) -> ()
  //   }) : (!stencil.temp<[1,65]xf64>) -> !stencil.temp<[0,64]xf64>
  //   "stencil.store"(%56, %50) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<[0,64]xf64>, !stencil.field<[-4,68]xf64>) -> ()
  //   func.return
  // }

  // func.func @stencil_two_stores(%59 : !stencil.field<[-4,68]xf64>, %60 : !stencil.field<[-4,68]xf64>, %61 : !stencil.field<[-4,68]xf64>) {
  //   %62 = "stencil.load"(%59) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[0,64]xf64>
  //   %63 = "stencil.apply"(%62) ({
  //   ^10(%64 : !stencil.temp<[0,64]xf64>):
  //     %65 = "stencil.access"(%64) {"offset" = #stencil.index<-1>} : (!stencil.temp<[0,64]xf64>) -> f64
  //     "stencil.return"(%65) : (f64) -> ()
  //   }) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[1,65]xf64>
  //   "stencil.store"(%63, %61) {"lb" = #stencil.index<1>, "ub" = #stencil.index<65>} : (!stencil.temp<[1,65]xf64>, !stencil.field<[-4,68]xf64>) -> ()
  //   %66 = "stencil.apply"(%63) ({
  //   ^11(%67 : !stencil.temp<[1,65]xf64>):
  //     %68 = "stencil.access"(%67) {"offset" = #stencil.index<1>} : (!stencil.temp<[1,65]xf64>) -> f64
  //     "stencil.return"(%68) : (f64) -> ()
  //   }) : (!stencil.temp<[1,65]xf64>) -> !stencil.temp<[0,64]xf64>
  //   "stencil.store"(%66, %60) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<[0,64]xf64>, !stencil.field<[-4,68]xf64>) -> ()
  //   func.return
  // }

  // func.func @apply_kernel(%69 : !stencil.field<[-2,13]x[-2,13]xf32>, %70 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
  //   %71 = "gpu.alloc"() {"operandSegmentSizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
  //   %u_vec_1 = "builtin.unrealized_conversion_cast"(%71) : (memref<15x15xf32>) -> !stencil.field<[-2,13]x[-2,13]xf32>
  //   %72 = "builtin.unrealized_conversion_cast"(%70) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> memref<15x15xf32>
  //   "gpu.memcpy"(%71, %72) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
  //   %73 = "gpu.alloc"() {"operandSegmentSizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
  //   %u_vec_0 = "builtin.unrealized_conversion_cast"(%73) : (memref<15x15xf32>) -> !stencil.field<[-2,13]x[-2,13]xf32>
  //   %74 = "builtin.unrealized_conversion_cast"(%69) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> memref<15x15xf32>
  //   "gpu.memcpy"(%73, %74) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
  //   %time_m_1 = arith.constant 0 : index
  //   %time_M_1 = arith.constant 10 : index
  //   %step_1 = arith.constant 1 : index
  //   %75, %76 = "scf.for"(%time_m_1, %time_M_1, %step_1, %u_vec_0, %u_vec_1) ({
  //   ^12(%time_1 : index, %t0 : !stencil.field<[-2,13]x[-2,13]xf32>, %t1 : !stencil.field<[-2,13]x[-2,13]xf32>):
  //     %t0_temp = "stencil.load"(%t0) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> !stencil.temp<[0,11]x[0,11]xf32>
  //     %t1_result = "stencil.apply"(%t0_temp) ({
  //     ^13(%t0_buff : !stencil.temp<[0,11]x[0,11]xf32>):
  //       %77 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[0,11]x[0,11]xf32>) -> f32
  //       "stencil.return"(%77) : (f32) -> ()
  //     }) : (!stencil.temp<[0,11]x[0,11]xf32>) -> !stencil.temp<[0,11]x[0,11]xf32>
  //     "stencil.store"(%t1_result, %t1) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<11, 11>} : (!stencil.temp<[0,11]x[0,11]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) -> ()
  //     "scf.yield"(%t1, %t0) : (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) -> ()
  //   }) : (index, index, index, !stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) -> (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>)
  //   func.return
  // }


  // func.func @stencil_init_float_unrolled(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
  //   %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  //   %3 = "stencil.apply"(%0) ({
  //   ^0(%4 : f64):
  //     %5 = arith.constant 1.0 : f64
  //     %6 = arith.constant 2.0 : f64
  //     %7 = arith.constant 3.0 : f64
  //     %8 = arith.constant 4.0 : f64
  //     %9 = arith.constant 5.0 : f64
  //     %10 = arith.constant 6.0 : f64
  //     %11 = arith.constant 7.0 : f64
  //     %12 = arith.constant 8.0 : f64
  //     "stencil.return"(%5, %6, %7, %8, %9, %10, %11, %12) <{"unroll" = #stencil.index<2, 2, 2>}> : (f64, f64, f64, f64, f64, f64, f64, f64) -> ()
  //   }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
  //   "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
  //   func.return

  // }


}
