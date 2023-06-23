// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir | filecheck %s

builtin.module {
// CHECK: builtin.module {

  // The pass used to crash on external function, just regression-testing this here.
  func.func @external(!stencil.field<?xf64>) -> ()
  // CHECK: func.func @external(memref<?xf64>) -> ()

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
  // CHECK:      func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
  // CHECK-NEXT:   %3 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
  // CHECK-NEXT:   %4 = arith.constant 1 : index
  // CHECK-NEXT:   %5 = arith.constant 2 : index
  // CHECK-NEXT:   %6 = arith.constant 3 : index
  // CHECK-NEXT:   %7 = arith.constant 1 : index
  // CHECK-NEXT:   %8 = arith.constant 65 : index
  // CHECK-NEXT:   %9 = arith.constant 66 : index
  // CHECK-NEXT:   %10 = arith.constant 63 : index
  // CHECK-NEXT:   "scf.parallel"(%4, %8, %7) ({
  // CHECK-NEXT:   ^0(%11 : index):
  // CHECK-NEXT:     "scf.for"(%5, %9, %7) ({
  // CHECK-NEXT:     ^1(%12 : index):
  // CHECK-NEXT:       "scf.for"(%6, %10, %7) ({
  // CHECK-NEXT:       ^2(%13 : index):
  // CHECK-NEXT:         %14 = arith.constant 1.0 : f64
  // CHECK-NEXT:         %15 = arith.addf %0, %14 : f64
  // CHECK-NEXT:         "memref.store"(%15, %3, %11, %12, %13) : (f64, memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>, index, index, index) -> ()
  // CHECK-NEXT:         "scf.yield"() : () -> ()
  // CHECK-NEXT:       }) : (index, index, index) -> ()
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @bufferswapping(%f0 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %f1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.field<[-2,2002]x[-2,2002]xf32> {
    %time_m = arith.constant 0 : index
    %time_M = arith.constant 1001 : index
    %step = arith.constant 1 : index
    %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
    ^1(%time : index, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %fi : !stencil.field<[-2,2002]x[-2,2002]xf32>):
      %tim1 = "stencil.load"(%fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.temp<[0,2000]x[0,2000]xf32>
      %ti = "stencil.apply"(%tim1) ({
      ^2(%tim1_b : !stencil.temp<[0,2000]x[0,2000]xf32>):
        %i = "stencil.access"(%tim1_b) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> f32
        "stencil.return"(%i) : (f32) -> ()
      }) : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> !stencil.temp<[0,2000]x[0,2000]xf32>
      "stencil.store"(%ti, %fi) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<2000, 2000>} : (!stencil.temp<[0,2000]x[0,2000]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
      "scf.yield"(%fi, %fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
    }) : (index, index, index, !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>)
    func.return %t1_out : !stencil.field<[-2,2002]x[-2,2002]xf32>
  }
  // CHECK:      func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
  // CHECK-NEXT:   %time_m = arith.constant 0 : index
  // CHECK-NEXT:   %time_M = arith.constant 1001 : index
  // CHECK-NEXT:   %step = arith.constant 1 : index
  // CHECK-NEXT:   %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
  // CHECK-NEXT:   ^3(%time : index, %fim1 : memref<2004x2004xf32>, %fi : memref<2004x2004xf32>):
  // CHECK-NEXT:     %fi_storeview = "memref.subview"(%fi) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
  // CHECK-NEXT:     %fim1_loadview = "memref.subview"(%fim1) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
  // CHECK-NEXT:     %16 = arith.constant 0 : index
  // CHECK-NEXT:     %17 = arith.constant 0 : index
  // CHECK-NEXT:     %18 = arith.constant 1 : index
  // CHECK-NEXT:     %19 = arith.constant 2000 : index
  // CHECK-NEXT:     %20 = arith.constant 2000 : index
  // CHECK-NEXT:     "scf.parallel"(%16, %19, %18) ({
  // CHECK-NEXT:     ^4(%21 : index):
  // CHECK-NEXT:       "scf.for"(%17, %20, %18) ({
  // CHECK-NEXT:       ^5(%22 : index):
  // CHECK-NEXT:         %i = "memref.load"(%fim1_loadview, %21, %22) : (memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> f32
  // CHECK-NEXT:         "memref.store"(%i, %fi_storeview, %21, %22) : (f32, memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> ()
  // CHECK-NEXT:         "scf.yield"() : () -> ()
  // CHECK-NEXT:       }) : (index, index, index) -> ()
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"(%fi, %fim1) : (memref<2004x2004xf32>, memref<2004x2004xf32>) -> ()
  // CHECK-NEXT:   }) : (index, index, index, memref<2004x2004xf32>, memref<2004x2004xf32>) -> (memref<2004x2004xf32>, memref<2004x2004xf32>)
  // CHECK-NEXT:   func.return %t1_out : memref<2004x2004xf32>
  // CHECK-NEXT: }

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
  // CHECK:      func.func @copy_1d(%23 : memref<?xf64>, %out : memref<?xf64>) {
  // CHECK-NEXT:   %24 = "memref.cast"(%23) : (memref<?xf64>) -> memref<72xf64>
  // CHECK-NEXT:   %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
  // CHECK-NEXT:   %outc_storeview = "memref.subview"(%outc) {"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
  // CHECK-NEXT:   %25 = "memref.subview"(%24) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:   %26 = arith.constant 0 : index
  // CHECK-NEXT:   %27 = arith.constant 1 : index
  // CHECK-NEXT:   %28 = arith.constant 68 : index
  // CHECK-NEXT:   "scf.parallel"(%26, %28, %27) ({
  // CHECK-NEXT:   ^6(%29 : index):
  // CHECK-NEXT:     %30 = arith.constant -1 : index
  // CHECK-NEXT:     %31 = arith.addi %29, %30 : index
  // CHECK-NEXT:     %32 = "memref.load"(%25, %31) : (memref<69xf64, strided<[1], offset: 4>>, index) -> f64
  // CHECK-NEXT:     "memref.store"(%32, %outc_storeview, %29) : (f64, memref<68xf64, strided<[1]>>, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @copy_2d(%0 : !stencil.field<?x?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,64]x[0,68]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,64]x[0,68]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,64]x[0,68]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,64]x[0,68]xf64>) -> !stencil.temp<[0,64]x[0,68]xf64>
    func.return
  }
  // CHECK:      func.func @copy_2d(%33 : memref<?x?xf64>) {
  // CHECK-NEXT:   %34 = "memref.cast"(%33) : (memref<?x?xf64>) -> memref<72x72xf64>
  // CHECK-NEXT:   %35 = "memref.subview"(%34) {"static_offsets" = array<i64: 4, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 292>>
  // CHECK-NEXT:   %36 = arith.constant 0 : index
  // CHECK-NEXT:   %37 = arith.constant 0 : index
  // CHECK-NEXT:   %38 = arith.constant 1 : index
  // CHECK-NEXT:   %39 = arith.constant 64 : index
  // CHECK-NEXT:   %40 = arith.constant 68 : index
  // CHECK-NEXT:   "scf.parallel"(%36, %39, %38) ({
  // CHECK-NEXT:   ^7(%41 : index):
  // CHECK-NEXT:     "scf.for"(%37, %40, %38) ({
  // CHECK-NEXT:     ^8(%42 : index):
  // CHECK-NEXT:       %43 = arith.constant -1 : index
  // CHECK-NEXT:       %44 = arith.addi %41, %43 : index
  // CHECK-NEXT:       %45 = "memref.load"(%35, %44, %42) : (memref<65x68xf64, strided<[72, 1], offset: 292>>, index, index) -> f64
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }


  func.func @copy_3d(%0 : !stencil.field<?x?x?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>) -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0, 1>} : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,68]xf64>
    func.return
  }
  // CHECK:       func.func @copy_3d(%46 : memref<?x?x?xf64>) {
  // CHECK-NEXT:    %47 = "memref.cast"(%46) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
  // CHECK-NEXT:    %48 = "memref.subview"(%47) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
  // CHECK-NEXT:    %49 = arith.constant 0 : index
  // CHECK-NEXT:    %50 = arith.constant 0 : index
  // CHECK-NEXT:    %51 = arith.constant 0 : index
  // CHECK-NEXT:    %52 = arith.constant 1 : index
  // CHECK-NEXT:    %53 = arith.constant 64 : index
  // CHECK-NEXT:    %54 = arith.constant 64 : index
  // CHECK-NEXT:    %55 = arith.constant 68 : index
  // CHECK-NEXT:    "scf.parallel"(%49, %53, %52) ({
  // CHECK-NEXT:    ^9(%56 : index):
  // CHECK-NEXT:      "scf.for"(%50, %54, %52) ({
  // CHECK-NEXT:      ^10(%57 : index):
  // CHECK-NEXT:        "scf.for"(%51, %55, %52) ({
  // CHECK-NEXT:        ^11(%58 : index):
  // CHECK-NEXT:          %59 = arith.constant -1 : index
  // CHECK-NEXT:          %60 = arith.addi %56, %59 : index
  // CHECK-NEXT:          %61 = arith.constant 1 : index
  // CHECK-NEXT:          %62 = arith.addi %58, %61 : index
  // CHECK-NEXT:          %63 = "memref.load"(%48, %60, %57, %62) : (memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>, index, index, index) -> f64
  // CHECK-NEXT:          "scf.yield"() : () -> ()
  // CHECK-NEXT:        }) : (index, index, index) -> ()
  // CHECK-NEXT:        "scf.yield"() : () -> ()
  // CHECK-NEXT:      }) : (index, index, index) -> ()
  // CHECK-NEXT:      "scf.yield"() : () -> ()
  // CHECK-NEXT:    }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:    func.return
  // CHECK-NEXT:  }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  // CHECK:      func.func @test_funcop_lowering(%64 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }
  // CHECK-NEXT: func.func @test_funcop_lowering_dyn(%65 : memref<8x8xf64>) {
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

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
  // CHECK:      func.func @offsets(%66 : memref<?x?x?xf64>, %67 : memref<?x?x?xf64>, %68 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   %69 = "memref.cast"(%66) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:   %70 = "memref.cast"(%67) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:   %71 = "memref.subview"(%70) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:   %72 = "memref.cast"(%68) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:   %73 = "memref.subview"(%69) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:   %74 = arith.constant 0 : index
  // CHECK-NEXT:   %75 = arith.constant 0 : index
  // CHECK-NEXT:   %76 = arith.constant 0 : index
  // CHECK-NEXT:   %77 = arith.constant 1 : index
  // CHECK-NEXT:   %78 = arith.constant 64 : index
  // CHECK-NEXT:   %79 = arith.constant 64 : index
  // CHECK-NEXT:   %80 = arith.constant 64 : index
  // CHECK-NEXT:   "scf.parallel"(%74, %78, %77) ({
  // CHECK-NEXT:   ^12(%81 : index):
  // CHECK-NEXT:     "scf.for"(%75, %79, %77) ({
  // CHECK-NEXT:     ^13(%82 : index):
  // CHECK-NEXT:       "scf.for"(%76, %80, %77) ({
  // CHECK-NEXT:       ^14(%83 : index):
  // CHECK-NEXT:         %84 = arith.constant -1 : index
  // CHECK-NEXT:         %85 = arith.addi %81, %84 : index
  // CHECK-NEXT:         %86 = "memref.load"(%73, %85, %82, %83) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
  // CHECK-NEXT:         %87 = arith.constant 1 : index
  // CHECK-NEXT:         %88 = arith.addi %81, %87 : index
  // CHECK-NEXT:         %89 = "memref.load"(%73, %88, %82, %83) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
  // CHECK-NEXT:         %90 = arith.constant 1 : index
  // CHECK-NEXT:         %91 = arith.addi %82, %90 : index
  // CHECK-NEXT:         %92 = "memref.load"(%73, %81, %91, %83) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
  // CHECK-NEXT:         %93 = arith.constant -1 : index
  // CHECK-NEXT:         %94 = arith.addi %82, %93 : index
  // CHECK-NEXT:         %95 = "memref.load"(%73, %81, %94, %83) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
  // CHECK-NEXT:         %96 = "memref.load"(%73, %81, %82, %83) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
  // CHECK-NEXT:         %97 = arith.addf %86, %89 : f64
  // CHECK-NEXT:         %98 = arith.addf %92, %95 : f64
  // CHECK-NEXT:         %99 = arith.addf %97, %98 : f64
  // CHECK-NEXT:         %cst = arith.constant -4.0 : f64
  // CHECK-NEXT:         %100 = arith.mulf %96, %cst : f64
  // CHECK-NEXT:         %101 = arith.addf %100, %99 : f64
  // CHECK-NEXT:         "memref.store"(%101, %71, %81, %82, %83) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
  // CHECK-NEXT:         "scf.yield"() : () -> ()
  // CHECK-NEXT:       }) : (index, index, index) -> ()
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
      "stencil.external_store"(%dyn_field, %dyn_mem) : (!stencil.field<?x?x?xf64>, memref<?x?x?xf64>) -> ()
      "stencil.external_store"(%sta_field, %sta_mem) : (!stencil.field<[-2,62]x[0,64]x[2,66]xf64>, memref<64x64x64xf64>) -> ()
      %0 = "stencil.external_load"(%dyn_mem) : (memref<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
      %1 = "stencil.external_load"(%sta_mem) : (memref<64x64x64xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>

      %casted = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
      func.return
  }
  // CHECK:       func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : memref<?x?x?xf64>, %sta_field : memref<64x64x64xf64>) {
  // CHECK-NEXT:    %casted = "memref.cast"(%dyn_mem) : (memref<?x?x?xf64>) -> memref<64x64x64xf64>
  // CHECK-NEXT:    func.return
  // CHECK-NEXT: }

  func.func @neg_bounds(%in : !stencil.field<[-32,32]xf64>, %out : !stencil.field<[-32,32]xf64>) {
    %tin = "stencil.load"(%in) : (!stencil.field<[-32,32]xf64>) -> !stencil.temp<[-16,16]xf64>
    %outt = "stencil.apply"(%tin) ({
    ^0(%tinb : !stencil.temp<[-16,16]xf64>):
      %val = "stencil.access"(%tinb) {"offset" = #stencil.index<0>} : (!stencil.temp<[-16,16]xf64>) -> f64
      "stencil.return"(%val) : (f64) -> ()
    }) : (!stencil.temp<[-16,16]xf64>) -> !stencil.temp<[-16,16]xf64>
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<-16>, "ub" = #stencil.index<16>} : (!stencil.temp<[-16,16]xf64>, !stencil.field<[-32,32]xf64>) -> ()
    func.return
  }
  // CHECK:      func.func @neg_bounds(%in : memref<64xf64>, %out_1 : memref<64xf64>) {
  // CHECK-NEXT:   %out_storeview = "memref.subview"(%out_1) {"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
  // CHECK-NEXT:   %in_loadview = "memref.subview"(%in) {"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
  // CHECK-NEXT:   %102 = arith.constant -16 : index
  // CHECK-NEXT:   %103 = arith.constant 1 : index
  // CHECK-NEXT:   %104 = arith.constant 16 : index
  // CHECK-NEXT:   "scf.parallel"(%102, %104, %103) ({
  // CHECK-NEXT:   ^15(%105 : index):
  // CHECK-NEXT:     %val = "memref.load"(%in_loadview, %105) : (memref<32xf64, strided<[1], offset: 32>>, index) -> f64
  // CHECK-NEXT:     "memref.store"(%val, %out_storeview, %105) : (f64, memref<32xf64, strided<[1], offset: 32>>, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @stencil_buffer(%49 : !stencil.field<[-4,68]xf64>, %50 : !stencil.field<[-4,68]xf64>) {
    %51 = "stencil.load"(%49) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[0,64]xf64>
    %52 = "stencil.apply"(%51) ({
    ^8(%53 : !stencil.temp<[0,64]xf64>):
      %54 = "stencil.access"(%53) {"offset" = #stencil.index<-1>} : (!stencil.temp<[0,64]xf64>) -> f64
      "stencil.return"(%54) : (f64) -> ()
    }) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[1,65]xf64>
    %55 = "stencil.buffer"(%52) : (!stencil.temp<[1,65]xf64>) -> !stencil.temp<[1,65]xf64>
    %56 = "stencil.apply"(%55) ({
    ^9(%57 : !stencil.temp<[1,65]xf64>):
      %58 = "stencil.access"(%57) {"offset" = #stencil.index<1>} : (!stencil.temp<[1,65]xf64>) -> f64
      "stencil.return"(%58) : (f64) -> ()
    }) : (!stencil.temp<[1,65]xf64>) -> !stencil.temp<[0,64]xf64>
    "stencil.store"(%56, %50) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<[0,64]xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }
  // CHECK:       func.func @stencil_buffer(%106 : memref<72xf64>, %107 : memref<72xf64>) {
  // CHECK-NEXT:    %108 = "memref.subview"(%107) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:    %109 = "memref.subview"(%106) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:    %110 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<64xf64>
  // CHECK-NEXT:    %111 = "memref.subview"(%110) {"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<64xf64, strided<[1], offset: -1>>
  // CHECK-NEXT:    %112 = arith.constant 1 : index
  // CHECK-NEXT:    %113 = arith.constant 1 : index
  // CHECK-NEXT:    %114 = arith.constant 65 : index
  // CHECK-NEXT:    "scf.parallel"(%112, %114, %113) ({
  // CHECK-NEXT:    ^16(%115 : index):
  // CHECK-NEXT:      %116 = arith.constant -1 : index
  // CHECK-NEXT:      %117 = arith.addi %115, %116 : index
  // CHECK-NEXT:      %118 = "memref.load"(%109, %117) : (memref<64xf64, strided<[1], offset: 4>>, index) -> f64
  // CHECK-NEXT:      "memref.store"(%118, %111, %115) : (f64, memref<64xf64, strided<[1], offset: -1>>, index) -> ()
  // CHECK-NEXT:      "scf.yield"() : () -> ()
  // CHECK-NEXT:    }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:    %119 = arith.constant 0 : index
  // CHECK-NEXT:    %120 = arith.constant 1 : index
  // CHECK-NEXT:    %121 = arith.constant 64 : index
  // CHECK-NEXT:    "scf.parallel"(%119, %121, %120) ({
  // CHECK-NEXT:    ^17(%122 : index):
  // CHECK-NEXT:      %123 = arith.constant 1 : index
  // CHECK-NEXT:      %124 = arith.addi %122, %123 : index
  // CHECK-NEXT:      %125 = "memref.load"(%111, %124) : (memref<64xf64, strided<[1], offset: -1>>, index) -> f64
  // CHECK-NEXT:      "memref.store"(%125, %108, %122) : (f64, memref<64xf64, strided<[1], offset: 4>>, index) -> ()
  // CHECK-NEXT:      "scf.yield"() : () -> ()
  // CHECK-NEXT:    }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:    "memref.dealloc"(%111) : (memref<64xf64, strided<[1], offset: -1>>) -> ()
  // CHECK-NEXT:    func.return
  // CHECK-NEXT:  }

  func.func @stencil_two_stores(%59 : !stencil.field<[-4,68]xf64>, %60 : !stencil.field<[-4,68]xf64>, %61 : !stencil.field<[-4,68]xf64>) {
    %62 = "stencil.load"(%59) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[0,64]xf64>
    %63 = "stencil.apply"(%62) ({
    ^10(%64 : !stencil.temp<[0,64]xf64>):
      %65 = "stencil.access"(%64) {"offset" = #stencil.index<-1>} : (!stencil.temp<[0,64]xf64>) -> f64
      "stencil.return"(%65) : (f64) -> ()
    }) : (!stencil.temp<[0,64]xf64>) -> !stencil.temp<[1,65]xf64>
    "stencil.store"(%63, %61) {"lb" = #stencil.index<1>, "ub" = #stencil.index<65>} : (!stencil.temp<[1,65]xf64>, !stencil.field<[-4,68]xf64>) -> ()
    %66 = "stencil.apply"(%63) ({
    ^11(%67 : !stencil.temp<[1,65]xf64>):
      %68 = "stencil.access"(%67) {"offset" = #stencil.index<1>} : (!stencil.temp<[1,65]xf64>) -> f64
      "stencil.return"(%68) : (f64) -> ()
    }) : (!stencil.temp<[1,65]xf64>) -> !stencil.temp<[0,64]xf64>
    "stencil.store"(%66, %60) {"lb" = #stencil.index<0>, "ub" = #stencil.index<64>} : (!stencil.temp<[0,64]xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }
  //CHECK:      func.func @stencil_two_stores(%126 : memref<72xf64>, %127 : memref<72xf64>, %128 : memref<72xf64>) {
  //CHECK-NEXT:   %129 = "memref.subview"(%127) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  //CHECK-NEXT:   %130 = "memref.subview"(%128) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  //CHECK-NEXT:   %131 = "memref.subview"(%126) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  //CHECK-NEXT:   %132 = arith.constant 1 : index
  //CHECK-NEXT:   %133 = arith.constant 1 : index
  //CHECK-NEXT:   %134 = arith.constant 65 : index
  //CHECK-NEXT:   "scf.parallel"(%132, %134, %133) ({
  //CHECK-NEXT:   ^18(%135 : index):
  //CHECK-NEXT:     %136 = arith.constant -1 : index
  //CHECK-NEXT:     %137 = arith.addi %135, %136 : index
  //CHECK-NEXT:     %138 = "memref.load"(%131, %137) : (memref<64xf64, strided<[1], offset: 4>>, index) -> f64
  //CHECK-NEXT:     "memref.store"(%138, %130, %135) : (f64, memref<64xf64, strided<[1], offset: 4>>, index) -> ()
  //CHECK-NEXT:     "scf.yield"() : () -> ()
  //CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  //CHECK-NEXT:   %139 = arith.constant 0 : index
  //CHECK-NEXT:   %140 = arith.constant 1 : index
  //CHECK-NEXT:   %141 = arith.constant 64 : index
  //CHECK-NEXT:   "scf.parallel"(%139, %141, %140) ({
  //CHECK-NEXT:   ^19(%142 : index):
  //CHECK-NEXT:     %143 = arith.constant 1 : index
  //CHECK-NEXT:     %144 = arith.addi %142, %143 : index
  //CHECK-NEXT:     %145 = "memref.load"(%130, %144) : (memref<64xf64, strided<[1], offset: 4>>, index) -> f64
  //CHECK-NEXT:     "memref.store"(%145, %129, %142) : (f64, memref<64xf64, strided<[1], offset: 4>>, index) -> ()
  //CHECK-NEXT:     "scf.yield"() : () -> ()
  //CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  //CHECK-NEXT:   func.return
  //CHECK-NEXT: }

}
// CHECK-NEXT: }
