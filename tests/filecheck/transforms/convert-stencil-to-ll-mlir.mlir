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
// CHECK-NEXT:   %3 = "memref.subview"(%2) {"static_offsets" = array<i64: 4, 5, 6>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 19956>>
// CHECK-NEXT:   %4 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %5 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:   %6 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   %7 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   %8 = "arith.constant"() {"value" = 60 : index} : () -> index
// CHECK-NEXT:   "scf.parallel"(%4, %6, %5) ({
// CHECK-NEXT:   ^0(%9 : index):
// CHECK-NEXT:     "scf.for"(%4, %7, %5) ({
// CHECK-NEXT:     ^1(%10 : index):
// CHECK-NEXT:       "scf.for"(%4, %8, %5) ({
// CHECK-NEXT:       ^2(%11 : index):
// CHECK-NEXT:         %12 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
// CHECK-NEXT:         %13 = arith.addf %0, %12 : f64
// CHECK-NEXT:         "memref.store"(%13, %3, %9, %10, %11) : (f64, memref<64x64x60xf64, strided<[4900, 70, 1], offset: 19956>>, index, index, index) -> ()
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
        %i = "stencil.access"(%tim1) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,2000]x[0,2000]xf32>) -> f32
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
// CHECK-NEXT:     %14 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %15 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %16 = "arith.constant"() {"value" = 2000 : index} : () -> index
// CHECK-NEXT:     %17 = "arith.constant"() {"value" = 2000 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%14, %16, %15) ({
// CHECK-NEXT:     ^4(%18 : index):
// CHECK-NEXT:       "scf.for"(%14, %17, %15) ({
// CHECK-NEXT:       ^5(%19 : index):
// CHECK-NEXT:         %i = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %i_1 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %i_2 = arith.addi %18, %i : index
// CHECK-NEXT:         %i_3 = arith.addi %19, %i_1 : index
// CHECK-NEXT:         %i_4 = "memref.load"(%fim1_loadview, %i_2, %i_3) : (memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> f32
// CHECK-NEXT:         "memref.store"(%i_4, %fi_storeview, %18, %19) : (f32, memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> ()
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

// CHECK:        func.func @copy_1d(%20 : memref<?xf64>, %out : memref<?xf64>) {
// CHECK-NEXT:     %21 = "memref.cast"(%20) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:     %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
// CHECK-NEXT:     %outc_storeview = "memref.subview"(%outc) {"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
// CHECK-NEXT:     %22 = "memref.subview"(%21) {"static_offsets" = array<i64: 3>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 3>>
// CHECK-NEXT:     %23 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %24 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %25 = "arith.constant"() {"value" = 68 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%23, %25, %24) ({
// CHECK-NEXT:     ^6(%26 : index):
// CHECK-NEXT:       %27 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %28 = arith.addi %26, %27 : index
// CHECK-NEXT:       %29 = "memref.load"(%22, %28) : (memref<69xf64, strided<[1], offset: 3>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%29, %outc_storeview, %26) : (f64, memref<68xf64, strided<[1]>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }

  func.func private @copy_2d(%0 : !stencil.field<?x?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,64]x[0,68]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,64]x[0,68]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,64]x[0,68]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,64]x[0,68]xf64>) -> !stencil.temp<[0,64]x[0,68]xf64>
    "func.return"() : () -> ()
  }
// CHECK:        func.func private @copy_2d(%30 : memref<?x?xf64>) {
// CHECK-NEXT:     %31 = "memref.cast"(%30) : (memref<?x?xf64>) -> memref<72x72xf64>
// CHECK-NEXT:     %32 = "memref.subview"(%31) {"static_offsets" = array<i64: 3, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 220>>
// CHECK-NEXT:     %33 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %34 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %35 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %36 = "arith.constant"() {"value" = 68 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%33, %35, %34) ({
// CHECK-NEXT:     ^7(%37 : index):
// CHECK-NEXT:       "scf.for"(%33, %36, %34) ({
// CHECK-NEXT:       ^8(%38 : index):
// CHECK-NEXT:         %39 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %40 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %41 = arith.addi %37, %39 : index
// CHECK-NEXT:         %42 = arith.addi %38, %40 : index
// CHECK-NEXT:         %43 = "memref.load"(%32, %41, %42) : (memref<65x68xf64, strided<[72, 1], offset: 220>>, index, index) -> f64
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }

  func.func private @copy_3d(%0 : !stencil.field<?x?x?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>) -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0, 1>} : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,68]xf64>
    "func.return"() : () -> ()
  }
// CHECK:       func.func private @copy_3d(%44 : memref<?x?x?xf64>) {
// CHECK-NEXT:    %45 = "memref.cast"(%44) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:    %46 = "memref.subview"(%45) {"static_offsets" = array<i64: 3, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 17180>>
// CHECK-NEXT:    %47 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:    %48 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:    %49 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:    %50 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:    %51 = "arith.constant"() {"value" = 68 : index} : () -> index
// CHECK-NEXT:    "scf.parallel"(%47, %49, %48) ({
// CHECK-NEXT:    ^9(%52 : index):
// CHECK-NEXT:      "scf.for"(%47, %50, %48) ({
// CHECK-NEXT:      ^10(%53 : index):
// CHECK-NEXT:        "scf.for"(%47, %51, %48) ({
// CHECK-NEXT:        ^11(%54 : index):
// CHECK-NEXT:          %55 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:          %56 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:          %57 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:          %58 = arith.addi %52, %55 : index
// CHECK-NEXT:          %59 = arith.addi %53, %56 : index
// CHECK-NEXT:          %60 = arith.addi %54, %57 : index
// CHECK-NEXT:          %61 = "memref.load"(%46, %58, %59, %60) : (memref<65x64x69xf64, strided<[5624, 76, 1], offset: 17180>>, index, index, index) -> f64
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
// CHECK:      func.func @test_funcop_lowering(%62 : memref<?x?x?xf64>) {
// CHECK-NEXT:   "func.return"() : () -> ()
// CHECK-NEXT: }

  func.func @test_funcop_lowering(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    "func.return"() : () -> ()
  }
// CHECK:      func.func @test_funcop_lowering(%63 : memref<8x8xf64>) {
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

// CHECK:      func.func @offsets(%64 : memref<?x?x?xf64>, %65 : memref<?x?x?xf64>, %66 : memref<?x?x?xf64>) {
// CHECK-NEXT:   %67 = "memref.cast"(%64) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %68 = "memref.cast"(%65) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %69 = "memref.subview"(%68) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:   %70 = "memref.cast"(%66) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %71 = "memref.subview"(%67) {"static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>
// CHECK-NEXT:   %72 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %73 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:   %74 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   %75 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   %76 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:   "scf.parallel"(%72, %74, %73) ({
// CHECK-NEXT:   ^12(%77 : index):
// CHECK-NEXT:     "scf.for"(%72, %75, %73) ({
// CHECK-NEXT:     ^13(%78 : index):
// CHECK-NEXT:       "scf.for"(%72, %76, %73) ({
// CHECK-NEXT:       ^14(%79 : index):
// CHECK-NEXT:         %80 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %81 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %82 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %83 = arith.addi %77, %80 : index
// CHECK-NEXT:         %84 = arith.addi %78, %81 : index
// CHECK-NEXT:         %85 = arith.addi %79, %82 : index
// CHECK-NEXT:         %86 = "memref.load"(%71, %83, %84, %85) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:         %87 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:         %88 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %89 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %90 = arith.addi %77, %87 : index
// CHECK-NEXT:         %91 = arith.addi %78, %88 : index
// CHECK-NEXT:         %92 = arith.addi %79, %89 : index
// CHECK-NEXT:         %93 = "memref.load"(%71, %90, %91, %92) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:         %94 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %95 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:         %96 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %97 = arith.addi %77, %94 : index
// CHECK-NEXT:         %98 = arith.addi %78, %95 : index
// CHECK-NEXT:         %99 = arith.addi %79, %96 : index
// CHECK-NEXT:         %100 = "memref.load"(%71, %97, %98, %99) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:         %101 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %102 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %103 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %104 = arith.addi %77, %101 : index
// CHECK-NEXT:         %105 = arith.addi %78, %102 : index
// CHECK-NEXT:         %106 = arith.addi %79, %103 : index
// CHECK-NEXT:         %107 = "memref.load"(%71, %104, %105, %106) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:         %108 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %109 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %110 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %111 = arith.addi %77, %108 : index
// CHECK-NEXT:         %112 = arith.addi %78, %109 : index
// CHECK-NEXT:         %113 = arith.addi %79, %110 : index
// CHECK-NEXT:         %114 = "memref.load"(%71, %111, %112, %113) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:         %115 = arith.addf %86, %93 : f64
// CHECK-NEXT:         %116 = arith.addf %100, %107 : f64
// CHECK-NEXT:         %117 = arith.addf %115, %116 : f64
// CHECK-NEXT:         %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:         %118 = arith.mulf %114, %cst : f64
// CHECK-NEXT:         %119 = arith.addf %118, %117 : f64
// CHECK-NEXT:         "memref.store"(%119, %69, %77, %78, %79) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
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
}
// CHECK:      func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : memref<?x?x?xf64>, %sta_field : memref<64x64x64xf64>) {
// CHECK-NEXT:    %casted = "memref.cast"(%dyn_mem) : (memref<?x?x?xf64>) -> memref<64x64x64xf64>
// CHECK-NEXT: }

}
// CHECK-NEXT: }
