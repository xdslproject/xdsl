// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir | filecheck %s

builtin.module {
// CHECK: builtin.module {

  func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
      %6 = arith.addf %4, %5 : f64
      "stencil.return"(%6) : (f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
    "func.return"() : () -> ()
  }

  // CHECK:      func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
  // CHECK-NEXT:   %3 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
  // CHECK-NEXT:   %4 = "arith.constant"() {"value" = 1 : index} : () -> index
  // CHECK-NEXT:   %5 = "arith.constant"() {"value" = 2 : index} : () -> index
  // CHECK-NEXT:   %6 = "arith.constant"() {"value" = 3 : index} : () -> index
  // CHECK-NEXT:   %7 = "arith.constant"() {"value" = 1 : index} : () -> index
  // CHECK-NEXT:   %8 = "arith.constant"() {"value" = 65 : index} : () -> index
  // CHECK-NEXT:   %9 = "arith.constant"() {"value" = 66 : index} : () -> index
  // CHECK-NEXT:   %10 = "arith.constant"() {"value" = 63 : index} : () -> index
  // CHECK-NEXT:   "scf.parallel"(%4, %8, %7) ({
  // CHECK-NEXT:   ^0(%11 : index):
  // CHECK-NEXT:     "scf.for"(%5, %9, %7) ({
  // CHECK-NEXT:     ^1(%12 : index):
  // CHECK-NEXT:       "scf.for"(%6, %10, %7) ({
  // CHECK-NEXT:       ^2(%13 : index):
  // CHECK-NEXT:         %14 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  // CHECK-NEXT:         %15 = arith.addf %0, %14 : f64
  // CHECK-NEXT:         "memref.store"(%15, %3, %11, %12, %13) : (f64, memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>, index, index, index) -> ()
  // CHECK-NEXT:         "scf.yield"() : () -> ()
  // CHECK-NEXT:       }) : (index, index, index) -> ()
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }

  func.func @bufferswapping(%f0 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %f1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.field<[-2,2002]x[-2,2002]xf32> {
    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %time_M = "arith.constant"() {"value" = 1001 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index
    %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
    ^1(%time : index, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %fi : !stencil.field<[-2,2002]x[-2,2002]xf32>):
      %tim1 = "stencil.load"(%fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.temp<[0,2000]x[0,2000]xf32>
      %ti = "stencil.apply"(%tim1) ({
      ^2(%tim1_b : !stencil.temp<[0,2000]x[0,2000]xf32>):
        %i = "stencil.access"(%tim1) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> f32
        "stencil.return"(%i) : (f32) -> ()
      }) : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> !stencil.temp<[0,2000]x[0,2000]xf32>
      "stencil.store"(%ti, %fi) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<2000, 2000>} : (!stencil.temp<[0,2000]x[0,2000]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
      "scf.yield"(%fi, %fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
    }) : (index, index, index, !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>)
    "func.return"(%t1_out) : (!stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
  }
// CHECK:      func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
// CHECK-NEXT:   %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %time_M = "arith.constant"() {"value" = 1001 : index} : () -> index
// CHECK-NEXT:   %step = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:   %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
// CHECK-NEXT:   ^3(%time : index, %fim1 : memref<2004x2004xf32>, %fi : memref<2004x2004xf32>):
// CHECK-NEXT:     %fi_storeview = "memref.subview"(%fi) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:     %fim1_loadview = "memref.subview"(%fim1) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:     %16 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %17 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %18 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %19 = "arith.constant"() {"value" = 2000 : index} : () -> index
// CHECK-NEXT:     %20 = "arith.constant"() {"value" = 2000 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%16, %19, %18) ({
// CHECK-NEXT:     ^4(%21 : index):
// CHECK-NEXT:       "scf.for"(%17, %20, %18) ({
// CHECK-NEXT:       ^5(%22 : index):
// CHECK-NEXT:         %i = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %i_1 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %i_2 = arith.addi %21, %i : index
// CHECK-NEXT:         %i_3 = arith.addi %22, %i_1 : index
// CHECK-NEXT:         %i_4 = "memref.load"(%fim1_loadview, %i_2, %i_3) : (memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> f32
// CHECK-NEXT:         "memref.store"(%i_4, %fi_storeview, %21, %22) : (f32, memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"(%fi, %fim1) : (memref<2004x2004xf32>, memref<2004x2004xf32>) -> ()
// CHECK-NEXT:   }) : (index, index, index, memref<2004x2004xf32>, memref<2004x2004xf32>) -> (memref<2004x2004xf32>, memref<2004x2004xf32>)
// CHECK-NEXT:   "func.return"(%t1_out) : (memref<2004x2004xf32>) -> ()
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
    "func.return"() : () -> ()
  }

  // CHECK:      func.func @copy_1d(%23 : memref<?xf64>, %out : memref<?xf64>) {
  // CHECK-NEXT:   %24 = "memref.cast"(%23) : (memref<?xf64>) -> memref<72xf64>
  // CHECK-NEXT:   %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
  // CHECK-NEXT:   %outc_storeview = "memref.subview"(%outc) {"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
  // CHECK-NEXT:   %25 = "memref.subview"(%24) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:   %26 = "arith.constant"() {"value" = 0 : index} : () -> index
  // CHECK-NEXT:   %27 = "arith.constant"() {"value" = 1 : index} : () -> index
  // CHECK-NEXT:   %28 = "arith.constant"() {"value" = 68 : index} : () -> index
  // CHECK-NEXT:   "scf.parallel"(%26, %28, %27) ({
  // CHECK-NEXT:   ^6(%29 : index):
  // CHECK-NEXT:     %30 = "arith.constant"() {"value" = -1 : index} : () -> index
  // CHECK-NEXT:     %31 = arith.addi %29, %30 : index
  // CHECK-NEXT:     %32 = "memref.load"(%25, %31) : (memref<69xf64, strided<[1], offset: 4>>, index) -> f64
  // CHECK-NEXT:     "memref.store"(%32, %outc_storeview, %29) : (f64, memref<68xf64, strided<[1]>>, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }

  func.func @copy_2d(%0 : !stencil.field<?x?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,64]x[0,68]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,64]x[0,68]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,64]x[0,68]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,64]x[0,68]xf64>) -> !stencil.temp<[0,64]x[0,68]xf64>
    "func.return"() : () -> ()
  }
  // CHECK:      func.func @copy_2d(%33 : memref<?x?xf64>) {
  // CHECK-NEXT:   %34 = "memref.cast"(%33) : (memref<?x?xf64>) -> memref<72x72xf64>
  // CHECK-NEXT:   %35 = "memref.subview"(%34) {"static_offsets" = array<i64: 4, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 292>>
  // CHECK-NEXT:   %36 = "arith.constant"() {"value" = 0 : index} : () -> index
  // CHECK-NEXT:   %37 = "arith.constant"() {"value" = 0 : index} : () -> index
  // CHECK-NEXT:   %38 = "arith.constant"() {"value" = 1 : index} : () -> index
  // CHECK-NEXT:   %39 = "arith.constant"() {"value" = 64 : index} : () -> index
  // CHECK-NEXT:   %40 = "arith.constant"() {"value" = 68 : index} : () -> index
  // CHECK-NEXT:   "scf.parallel"(%36, %39, %38) ({
  // CHECK-NEXT:   ^7(%41 : index):
  // CHECK-NEXT:     "scf.for"(%37, %40, %38) ({
  // CHECK-NEXT:     ^8(%42 : index):
  // CHECK-NEXT:       %43 = "arith.constant"() {"value" = -1 : index} : () -> index
  // CHECK-NEXT:       %44 = "arith.constant"() {"value" = 0 : index} : () -> index
  // CHECK-NEXT:       %45 = arith.addi %41, %43 : index
  // CHECK-NEXT:       %46 = arith.addi %42, %44 : index
  // CHECK-NEXT:       %47 = "memref.load"(%35, %45, %46) : (memref<65x68xf64, strided<[72, 1], offset: 292>>, index, index) -> f64
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }

  func.func @copy_3d(%0 : !stencil.field<?x?x?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>) -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0, 1>} : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,68]xf64>
    "func.return"() : () -> ()
  }
// CHECK:       func.func @copy_3d(%48 : memref<?x?x?xf64>) {
// CHECK-NEXT:    %49 = "memref.cast"(%48) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:    %50 = "memref.subview"(%49) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:    %51 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:    %52 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:    %53 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:    %54 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:    %55 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:    %56 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:    %57 = "arith.constant"() {"value" = 68 : index} : () -> index
// CHECK-NEXT:    "scf.parallel"(%51, %55, %54) ({
// CHECK-NEXT:    ^9(%58 : index):
// CHECK-NEXT:      "scf.for"(%52, %56, %54) ({
// CHECK-NEXT:      ^10(%59 : index):
// CHECK-NEXT:        "scf.for"(%53, %57, %54) ({
// CHECK-NEXT:        ^11(%60 : index):
// CHECK-NEXT:          %61 = "arith.constant"() {"value" = -1 : index} : () -> index
// CHECK-NEXT:          %62 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:          %63 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:          %64 = arith.addi %58, %61 : index
// CHECK-NEXT:          %65 = arith.addi %59, %62 : index
// CHECK-NEXT:          %66 = arith.addi %60, %63 : index
// CHECK-NEXT:          %67 = "memref.load"(%50, %64, %65, %66) : (memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>, index, index, index) -> f64
// CHECK-NEXT:          "scf.yield"() : () -> ()
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "scf.yield"() : () -> ()
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "scf.yield"() : () -> ()
// CHECK-NEXT:    }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT:  }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    "func.return"() : () -> ()
  }
  // CHECK:      func.func @test_funcop_lowering(%68 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }

  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    "func.return"() : () -> ()
  }
  // CHECK-NEXT: func.func @test_funcop_lowering_dyn(%69 : memref<8x8xf64>) {
  // CHECK-NEXT:   "func.return"() : () -> ()
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
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = arith.mulf %14, %cst : f64
      %19 = arith.addf %18, %17 : f64
      "stencil.return"(%19, %18) : (f64, f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<?x?x?xf64>)
    "stencil.store"(%7, %4) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }

// CHECK:      func.func @offsets(%70 : memref<?x?x?xf64>, %71 : memref<?x?x?xf64>, %72 : memref<?x?x?xf64>) {
// CHECK-NEXT:   %73 = "memref.cast"(%70) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %74 = "memref.cast"(%71) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %75 = "memref.subview"(%74) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:   %76 = "memref.cast"(%72) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %77 = "memref.subview"(%73) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:   %78 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %79 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %80 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %81 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:   %82 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   %83 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   %84 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   "scf.parallel"(%78, %82, %81) ({
// CHECK-NEXT:   ^12(%85 : index):
// CHECK-NEXT:     "scf.for"(%79, %83, %81) ({
// CHECK-NEXT:     ^13(%86 : index):
// CHECK-NEXT:       "scf.for"(%80, %84, %81) ({
// CHECK-NEXT:       ^14(%87 : index):
// CHECK-NEXT:         %88 = "arith.constant"() {"value" = -1 : index} : () -> index
// CHECK-NEXT:         %89 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %90 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %91 = arith.addi %85, %88 : index
// CHECK-NEXT:         %92 = arith.addi %86, %89 : index
// CHECK-NEXT:         %93 = arith.addi %87, %90 : index
// CHECK-NEXT:         %94 = "memref.load"(%77, %91, %92, %93) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:         %95 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %96 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %97 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %98 = arith.addi %85, %95 : index
// CHECK-NEXT:         %99 = arith.addi %86, %96 : index
// CHECK-NEXT:         %100 = arith.addi %87, %97 : index
// CHECK-NEXT:         %101 = "memref.load"(%77, %98, %99, %100) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:         %102 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %103 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %104 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %105 = arith.addi %85, %102 : index
// CHECK-NEXT:         %106 = arith.addi %86, %103 : index
// CHECK-NEXT:         %107 = arith.addi %87, %104 : index
// CHECK-NEXT:         %108 = "memref.load"(%77, %105, %106, %107) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:         %109 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %110 = "arith.constant"() {"value" = -1 : index} : () -> index
// CHECK-NEXT:         %111 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %112 = arith.addi %85, %109 : index
// CHECK-NEXT:         %113 = arith.addi %86, %110 : index
// CHECK-NEXT:         %114 = arith.addi %87, %111 : index
// CHECK-NEXT:         %115 = "memref.load"(%77, %112, %113, %114) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:         %116 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %117 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %118 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %119 = arith.addi %85, %116 : index
// CHECK-NEXT:         %120 = arith.addi %86, %117 : index
// CHECK-NEXT:         %121 = arith.addi %87, %118 : index
// CHECK-NEXT:         %122 = "memref.load"(%77, %119, %120, %121) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:         %123 = arith.addf %94, %101 : f64
// CHECK-NEXT:         %124 = arith.addf %108, %115 : f64
// CHECK-NEXT:         %125 = arith.addf %123, %124 : f64
// CHECK-NEXT:         %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:         %126 = arith.mulf %122, %cst : f64
// CHECK-NEXT:         %127 = arith.addf %126, %125 : f64
// CHECK-NEXT:         "memref.store"(%127, %75, %85, %86, %87) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   "func.return"() : () -> ()
// CHECK-NEXT: }

func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
    "stencil.external_store"(%dyn_field, %dyn_mem) : (!stencil.field<?x?x?xf64>, memref<?x?x?xf64>) -> ()
    "stencil.external_store"(%sta_field, %sta_mem) : (!stencil.field<[-2,62]x[0,64]x[2,66]xf64>, memref<64x64x64xf64>) -> ()
    %0 = "stencil.external_load"(%dyn_mem) : (memref<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
    %1 = "stencil.external_load"(%sta_mem) : (memref<64x64x64xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>

    %casted = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
    "func.return"() : () -> ()
}
// CHECK:      func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : memref<?x?x?xf64>, %sta_field : memref<64x64x64xf64>) {
// CHECK-NEXT:    %casted = "memref.cast"(%dyn_mem) : (memref<?x?x?xf64>) -> memref<64x64x64xf64>
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT: }

func.func @neg_bounds(%in : !stencil.field<[-32,32]xf64>, %out : !stencil.field<[-32,32]xf64>) {
  %tin = "stencil.load"(%in) : (!stencil.field<[-32,32]xf64>) -> !stencil.temp<[-16,16]xf64>
  %outt = "stencil.apply"(%tin) ({
  ^0(%tinb : !stencil.temp<[-16,16]xf64>):
    %val = "stencil.access"(%tinb) {"offset" = #stencil.index<0>} : (!stencil.temp<[-16,16]xf64>) -> f64
    "stencil.return"(%val) : (f64) -> ()
  }) : (!stencil.temp<[-16,16]xf64>) -> !stencil.temp<[-16,16]xf64>
  "stencil.store"(%outt, %out) {"lb" = #stencil.index<-16>, "ub" = #stencil.index<16>} : (!stencil.temp<[-16,16]xf64>, !stencil.field<[-32,32]xf64>) -> ()
  "func.return"() : () -> ()
}
// CHECK:      func.func @neg_bounds(%in : memref<64xf64>, %out_1 : memref<64xf64>) {
// CHECK-NEXT:   %out_storeview = "memref.subview"(%out_1) {"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:   %in_loadview = "memref.subview"(%in) {"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:   %128 = "arith.constant"() {"value" = -16 : index} : () -> index
// CHECK-NEXT:   %129 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:   %130 = "arith.constant"() {"value" = 16 : index} : () -> index
// CHECK-NEXT:   "scf.parallel"(%128, %130, %129) ({
// CHECK-NEXT:   ^15(%131 : index):
// CHECK-NEXT:     %val = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %val_1 = arith.addi %131, %val : index
// CHECK-NEXT:     %val_2 = "memref.load"(%in_loadview, %val_1) : (memref<32xf64, strided<[1], offset: 32>>, index) -> f64
// CHECK-NEXT:     "memref.store"(%val_2, %out_storeview, %131) : (f64, memref<32xf64, strided<[1], offset: 32>>, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   "func.return"() : () -> ()
// CHECK-NEXT: }

}
// CHECK-NEXT: }
