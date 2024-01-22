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
// CHECK:         func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:      %3 = "memref.subview"(%2) <{"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 3 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      %9 = arith.constant 1 : index
// CHECK-NEXT:      %10 = arith.constant 1 : index
// CHECK-NEXT:      %11 = arith.constant 65 : index
// CHECK-NEXT:      %12 = arith.constant 66 : index
// CHECK-NEXT:      %13 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%4, %11, %8) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^0(%14 : index):
// CHECK-NEXT:        scf.for %15 = %5 to %12 step %9 {
// CHECK-NEXT:          scf.for %16 = %6 to %13 step %10 {
// CHECK-NEXT:            %17 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:            %18 = arith.addf %0, %17 : f64
// CHECK-NEXT:            memref.store %18, %3[%14, %15, %16] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

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
// CHECK:         func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1001 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %t1_out, %t0_out = scf.for %time = %time_m to %time_M step %step iter_args(%fim1 = %f0, %fi = %f1) -> (memref<2004x2004xf32>, memref<2004x2004xf32>) {
// CHECK-NEXT:        %fi_storeview = "memref.subview"(%fi) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:        %fim1_loadview = "memref.subview"(%fim1) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:        %19 = arith.constant 0 : index
// CHECK-NEXT:        %20 = arith.constant 0 : index
// CHECK-NEXT:        %21 = arith.constant 1 : index
// CHECK-NEXT:        %22 = arith.constant 1 : index
// CHECK-NEXT:        %23 = arith.constant 1 : index
// CHECK-NEXT:        %24 = arith.constant 2000 : index
// CHECK-NEXT:        %25 = arith.constant 2000 : index
// CHECK-NEXT:        "scf.parallel"(%19, %24, %22) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:        ^1(%26 : index):
// CHECK-NEXT:          scf.for %27 = %20 to %25 step %23 {
// CHECK-NEXT:            %i = memref.load %fim1_loadview[%26, %27] : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:            memref.store %i, %fi_storeview[%26, %27] : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        scf.yield %fi, %fim1 : memref<2004x2004xf32>, memref<2004x2004xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %t1_out : memref<2004x2004xf32>
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
// CHECK:         func.func @copy_1d(%28 : memref<?xf64>, %out : memref<?xf64>) {
// CHECK-NEXT:      %29 = "memref.cast"(%28) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:      %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
// CHECK-NEXT:      %outc_storeview = "memref.subview"(%outc) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
// CHECK-NEXT:      %30 = "memref.subview"(%29) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %31 = arith.constant 0 : index
// CHECK-NEXT:      %32 = arith.constant 1 : index
// CHECK-NEXT:      %33 = arith.constant 1 : index
// CHECK-NEXT:      %34 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%31, %34, %33) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^2(%35 : index):
// CHECK-NEXT:        %36 = arith.constant -1 : index
// CHECK-NEXT:        %37 = arith.addi %35, %36 : index
// CHECK-NEXT:        %38 = memref.load %30[%37] : memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %38, %outc_storeview[%35] : memref<68xf64, strided<[1]>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

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
// CHECK:         func.func @copy_2d(%39 : memref<?x?xf64>) {
// CHECK-NEXT:      %40 = "memref.cast"(%39) : (memref<?x?xf64>) -> memref<72x72xf64>
// CHECK-NEXT:      %41 = "memref.subview"(%40) <{"static_offsets" = array<i64: 4, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:      %42 = arith.constant 0 : index
// CHECK-NEXT:      %43 = arith.constant 0 : index
// CHECK-NEXT:      %44 = arith.constant 1 : index
// CHECK-NEXT:      %45 = arith.constant 1 : index
// CHECK-NEXT:      %46 = arith.constant 1 : index
// CHECK-NEXT:      %47 = arith.constant 64 : index
// CHECK-NEXT:      %48 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%42, %47, %45) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^3(%49 : index):
// CHECK-NEXT:        scf.for %50 = %43 to %48 step %46 {
// CHECK-NEXT:          %51 = arith.constant -1 : index
// CHECK-NEXT:          %52 = arith.addi %49, %51 : index
// CHECK-NEXT:          %53 = memref.load %41[%52, %50] : memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


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
// CHECK:         func.func @copy_3d(%54 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %55 = "memref.cast"(%54) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:      %56 = "memref.subview"(%55) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:      %57 = arith.constant 0 : index
// CHECK-NEXT:      %58 = arith.constant 0 : index
// CHECK-NEXT:      %59 = arith.constant 0 : index
// CHECK-NEXT:      %60 = arith.constant 1 : index
// CHECK-NEXT:      %61 = arith.constant 1 : index
// CHECK-NEXT:      %62 = arith.constant 1 : index
// CHECK-NEXT:      %63 = arith.constant 1 : index
// CHECK-NEXT:      %64 = arith.constant 64 : index
// CHECK-NEXT:      %65 = arith.constant 64 : index
// CHECK-NEXT:      %66 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%57, %64, %61) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^4(%67 : index):
// CHECK-NEXT:        scf.for %68 = %58 to %65 step %62 {
// CHECK-NEXT:          scf.for %69 = %59 to %66 step %63 {
// CHECK-NEXT:            %70 = arith.constant -1 : index
// CHECK-NEXT:            %71 = arith.addi %67, %70 : index
// CHECK-NEXT:            %72 = arith.constant 1 : index
// CHECK-NEXT:            %73 = arith.addi %69, %72 : index
// CHECK-NEXT:            %74 = memref.load %56[%71, %68, %73] : memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }
// CHECK:       func.func @test_funcop_lowering(%75 : memref<?x?x?xf64>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @test_funcop_lowering_dyn(%76 : memref<8x8xf64>) {
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
// CHECK:         func.func @offsets(%77 : memref<?x?x?xf64>, %78 : memref<?x?x?xf64>, %79 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %80 = "memref.cast"(%77) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %81 = "memref.cast"(%78) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %82 = "memref.subview"(%81) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %83 = "memref.cast"(%79) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %84 = "memref.subview"(%80) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %85 = arith.constant 0 : index
// CHECK-NEXT:      %86 = arith.constant 0 : index
// CHECK-NEXT:      %87 = arith.constant 0 : index
// CHECK-NEXT:      %88 = arith.constant 1 : index
// CHECK-NEXT:      %89 = arith.constant 1 : index
// CHECK-NEXT:      %90 = arith.constant 1 : index
// CHECK-NEXT:      %91 = arith.constant 1 : index
// CHECK-NEXT:      %92 = arith.constant 64 : index
// CHECK-NEXT:      %93 = arith.constant 64 : index
// CHECK-NEXT:      %94 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%85, %92, %89) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^5(%95 : index):
// CHECK-NEXT:        scf.for %96 = %86 to %93 step %90 {
// CHECK-NEXT:          scf.for %97 = %87 to %94 step %91 {
// CHECK-NEXT:            %98 = arith.constant -1 : index
// CHECK-NEXT:            %99 = arith.addi %95, %98 : index
// CHECK-NEXT:            %100 = memref.load %84[%99, %96, %97] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %101 = arith.constant 1 : index
// CHECK-NEXT:            %102 = arith.addi %95, %101 : index
// CHECK-NEXT:            %103 = memref.load %84[%102, %96, %97] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %104 = arith.constant 1 : index
// CHECK-NEXT:            %105 = arith.addi %96, %104 : index
// CHECK-NEXT:            %106 = memref.load %84[%95, %105, %97] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %107 = arith.constant -1 : index
// CHECK-NEXT:            %108 = arith.addi %96, %107 : index
// CHECK-NEXT:            %109 = memref.load %84[%95, %108, %97] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %110 = memref.load %84[%95, %96, %97] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:            %111 = arith.addf %100, %103 : f64
// CHECK-NEXT:            %112 = arith.addf %106, %109 : f64
// CHECK-NEXT:            %113 = arith.addf %111, %112 : f64
// CHECK-NEXT:            %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:            %114 = arith.mulf %110, %cst : f64
// CHECK-NEXT:            %115 = arith.addf %114, %113 : f64
// CHECK-NEXT:            memref.store %115, %82[%95, %96, %97] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
      "stencil.external_store"(%dyn_field, %dyn_mem) : (!stencil.field<?x?x?xf64>, memref<?x?x?xf64>) -> ()
      "stencil.external_store"(%sta_field, %sta_mem) : (!stencil.field<[-2,62]x[0,64]x[2,66]xf64>, memref<64x64x64xf64>) -> ()
      %0 = "stencil.external_load"(%dyn_mem) : (memref<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
      %1 = "stencil.external_load"(%sta_mem) : (memref<64x64x64xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>

      %casted = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
      func.return
  }
// CHECK:         func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : memref<?x?x?xf64>, %sta_field : memref<64x64x64xf64>) {
// CHECK-NEXT:      %casted = "memref.cast"(%dyn_mem) : (memref<?x?x?xf64>) -> memref<64x64x64xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

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
// CHECK:         func.func @neg_bounds(%in : memref<64xf64>, %out_1 : memref<64xf64>) {
// CHECK-NEXT:      %out_storeview = "memref.subview"(%out_1) <{"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:      %in_loadview = "memref.subview"(%in) <{"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:      %116 = arith.constant -16 : index
// CHECK-NEXT:      %117 = arith.constant 1 : index
// CHECK-NEXT:      %118 = arith.constant 1 : index
// CHECK-NEXT:      %119 = arith.constant 16 : index
// CHECK-NEXT:      "scf.parallel"(%116, %119, %118) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^6(%120 : index):
// CHECK-NEXT:        %val = memref.load %in_loadview[%120] : memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:        memref.store %val, %out_storeview[%120] : memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

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
// CHECK:         func.func @stencil_buffer(%121 : memref<72xf64>, %122 : memref<72xf64>) {
// CHECK-NEXT:      %123 = "memref.subview"(%122) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %124 = "memref.subview"(%121) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %125 = "memref.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64xf64>
// CHECK-NEXT:      %126 = "memref.subview"(%125) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64xf64>) -> memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:      %127 = arith.constant 1 : index
// CHECK-NEXT:      %128 = arith.constant 1 : index
// CHECK-NEXT:      %129 = arith.constant 1 : index
// CHECK-NEXT:      %130 = arith.constant 65 : index
// CHECK-NEXT:      "scf.parallel"(%127, %130, %129) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^7(%131 : index):
// CHECK-NEXT:        %132 = arith.constant -1 : index
// CHECK-NEXT:        %133 = arith.addi %131, %132 : index
// CHECK-NEXT:        %134 = memref.load %124[%133] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %134, %126[%131] : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      %135 = arith.constant 0 : index
// CHECK-NEXT:      %136 = arith.constant 1 : index
// CHECK-NEXT:      %137 = arith.constant 1 : index
// CHECK-NEXT:      %138 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%135, %138, %137) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^8(%139 : index):
// CHECK-NEXT:        %140 = arith.constant 1 : index
// CHECK-NEXT:        %141 = arith.addi %139, %140 : index
// CHECK-NEXT:        %142 = memref.load %126[%141] : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:        memref.store %142, %123[%139] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "memref.dealloc"(%126) : (memref<64xf64, strided<[1], offset: -1>>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

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
// CHECK:         func.func @stencil_two_stores(%143 : memref<72xf64>, %144 : memref<72xf64>, %145 : memref<72xf64>) {
// CHECK-NEXT:      %146 = "memref.subview"(%144) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %147 = "memref.subview"(%145) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %148 = "memref.subview"(%143) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %149 = arith.constant 1 : index
// CHECK-NEXT:      %150 = arith.constant 1 : index
// CHECK-NEXT:      %151 = arith.constant 1 : index
// CHECK-NEXT:      %152 = arith.constant 65 : index
// CHECK-NEXT:      "scf.parallel"(%149, %152, %151) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^9(%153 : index):
// CHECK-NEXT:        %154 = arith.constant -1 : index
// CHECK-NEXT:        %155 = arith.addi %153, %154 : index
// CHECK-NEXT:        %156 = memref.load %148[%155] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %156, %147[%153] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      %157 = arith.constant 0 : index
// CHECK-NEXT:      %158 = arith.constant 1 : index
// CHECK-NEXT:      %159 = arith.constant 1 : index
// CHECK-NEXT:      %160 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%157, %160, %159) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^10(%161 : index):
// CHECK-NEXT:        %162 = arith.constant 1 : index
// CHECK-NEXT:        %163 = arith.addi %161, %162 : index
// CHECK-NEXT:        %164 = memref.load %147[%163] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %164, %146[%161] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @apply_kernel(%69 : !stencil.field<[-2,13]x[-2,13]xf32>, %70 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
    %71 = "gpu.alloc"() {"operandSegmentSizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
    %u_vec_1 = "builtin.unrealized_conversion_cast"(%71) : (memref<15x15xf32>) -> !stencil.field<[-2,13]x[-2,13]xf32>
    %72 = "builtin.unrealized_conversion_cast"(%70) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> memref<15x15xf32>
    "gpu.memcpy"(%71, %72) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %73 = "gpu.alloc"() {"operandSegmentSizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
    %u_vec_0 = "builtin.unrealized_conversion_cast"(%73) : (memref<15x15xf32>) -> !stencil.field<[-2,13]x[-2,13]xf32>
    %74 = "builtin.unrealized_conversion_cast"(%69) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> memref<15x15xf32>
    "gpu.memcpy"(%73, %74) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %time_m_1 = arith.constant 0 : index
    %time_M_1 = arith.constant 10 : index
    %step_1 = arith.constant 1 : index
    %75, %76 = "scf.for"(%time_m_1, %time_M_1, %step_1, %u_vec_0, %u_vec_1) ({
    ^12(%time_1 : index, %t0 : !stencil.field<[-2,13]x[-2,13]xf32>, %t1 : !stencil.field<[-2,13]x[-2,13]xf32>):
      %t0_temp = "stencil.load"(%t0) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> !stencil.temp<[0,11]x[0,11]xf32>
      %t1_result = "stencil.apply"(%t0_temp) ({
      ^13(%t0_buff : !stencil.temp<[0,11]x[0,11]xf32>):
        %77 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[0,11]x[0,11]xf32>) -> f32
        "stencil.return"(%77) : (f32) -> ()
      }) : (!stencil.temp<[0,11]x[0,11]xf32>) -> !stencil.temp<[0,11]x[0,11]xf32>
      "stencil.store"(%t1_result, %t1) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<11, 11>} : (!stencil.temp<[0,11]x[0,11]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) -> ()
      "scf.yield"(%t1, %t0) : (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) -> ()
    }) : (index, index, index, !stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) -> (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>)
    func.return
  }

// CHECK:         func.func @apply_kernel(%165 : memref<15x15xf32>, %166 : memref<15x15xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
// CHECK-NEXT:      %167 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec_1 = builtin.unrealized_conversion_cast %167 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      %168 = builtin.unrealized_conversion_cast %166 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%167, %168) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %169 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec_0 = builtin.unrealized_conversion_cast %169 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      %170 = builtin.unrealized_conversion_cast %165 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%169, %170) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %time_m_1 = arith.constant 0 : index
// CHECK-NEXT:      %time_M_1 = arith.constant 10 : index
// CHECK-NEXT:      %step_1 = arith.constant 1 : index
// CHECK-NEXT:      %171, %172 = scf.for %time_1 = %time_m_1 to %time_M_1 step %step_1 iter_args(%t0 = %u_vec_0, %t1 = %u_vec_1) -> (memref<15x15xf32>, memref<15x15xf32>) {
// CHECK-NEXT:        %t1_storeview = "memref.subview"(%t1) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<15x15xf32>) -> memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:        %t0_loadview = "memref.subview"(%t0) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<15x15xf32>) -> memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:        %173 = arith.constant 0 : index
// CHECK-NEXT:        %174 = arith.constant 0 : index
// CHECK-NEXT:        %175 = arith.constant 1 : index
// CHECK-NEXT:        %176 = arith.constant 1 : index
// CHECK-NEXT:        %177 = arith.constant 1 : index
// CHECK-NEXT:        %178 = arith.constant 11 : index
// CHECK-NEXT:        %179 = arith.constant 11 : index
// CHECK-NEXT:        "scf.parallel"(%173, %178, %176) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:        ^11(%180 : index):
// CHECK-NEXT:          scf.for %181 = %174 to %179 step %177 {
// CHECK-NEXT:            %182 = memref.load %t0_loadview[%180, %181] : memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:            memref.store %182, %t1_storeview[%180, %181] : memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        scf.yield %t1, %t0 : memref<15x15xf32>, memref<15x15xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_float_unrolled(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = arith.constant 1.0 : f64
      %6 = arith.constant 2.0 : f64
      %7 = arith.constant 3.0 : f64
      %8 = arith.constant 4.0 : f64
      %9 = arith.constant 5.0 : f64
      %10 = arith.constant 6.0 : f64
      %11 = arith.constant 7.0 : f64
      %12 = arith.constant 8.0 : f64
      "stencil.return"(%5, %6, %7, %8, %9, %10, %11, %12) <{"unroll" = #stencil.index<2, 2, 2>}> : (f64, f64, f64, f64, f64, f64, f64, f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
    func.return

  }

// CHECK-NEXT:    func.func @stencil_init_float_unrolled(%183 : f64, %184 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %185 = "memref.cast"(%184) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:      %186 = "memref.subview"(%185) <{"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:      %187 = arith.constant 1 : index
// CHECK-NEXT:      %188 = arith.constant 2 : index
// CHECK-NEXT:      %189 = arith.constant 3 : index
// CHECK-NEXT:      %190 = arith.constant 1 : index
// CHECK-NEXT:      %191 = arith.constant 2 : index
// CHECK-NEXT:      %192 = arith.constant 2 : index
// CHECK-NEXT:      %193 = arith.constant 2 : index
// CHECK-NEXT:      %194 = arith.constant 65 : index
// CHECK-NEXT:      %195 = arith.constant 66 : index
// CHECK-NEXT:      %196 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%187, %194, %191) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^12(%197 : index):
// CHECK-NEXT:        scf.for %198 = %188 to %195 step %192 {
// CHECK-NEXT:          scf.for %199 = %189 to %196 step %193 {
// CHECK-NEXT:            %200 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:            %201 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:            %202 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:            %203 = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:            %204 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:            %205 = arith.constant 6.000000e+00 : f64
// CHECK-NEXT:            %206 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:            %207 = arith.constant 8.000000e+00 : f64
// CHECK-NEXT:            memref.store %200, %186[%197, %198, %199] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %208 = arith.constant 1 : index
// CHECK-NEXT:            %209 = arith.addi %199, %208 : index
// CHECK-NEXT:            memref.store %201, %186[%197, %198, %209] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %210 = arith.constant 1 : index
// CHECK-NEXT:            %211 = arith.addi %198, %210 : index
// CHECK-NEXT:            memref.store %202, %186[%197, %211, %199] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %212 = arith.constant 1 : index
// CHECK-NEXT:            %213 = arith.addi %198, %212 : index
// CHECK-NEXT:            %214 = arith.constant 1 : index
// CHECK-NEXT:            %215 = arith.addi %199, %214 : index
// CHECK-NEXT:            memref.store %203, %186[%197, %213, %215] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %216 = arith.constant 1 : index
// CHECK-NEXT:            %217 = arith.addi %197, %216 : index
// CHECK-NEXT:            memref.store %204, %186[%217, %198, %199] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %218 = arith.constant 1 : index
// CHECK-NEXT:            %219 = arith.addi %197, %218 : index
// CHECK-NEXT:            %220 = arith.constant 1 : index
// CHECK-NEXT:            %221 = arith.addi %199, %220 : index
// CHECK-NEXT:            memref.store %205, %186[%219, %198, %221] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %222 = arith.constant 1 : index
// CHECK-NEXT:            %223 = arith.addi %197, %222 : index
// CHECK-NEXT:            %224 = arith.constant 1 : index
// CHECK-NEXT:            %225 = arith.addi %198, %224 : index
// CHECK-NEXT:            memref.store %206, %186[%223, %225, %199] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:            %226 = arith.constant 1 : index
// CHECK-NEXT:            %227 = arith.addi %197, %226 : index
// CHECK-NEXT:            %228 = arith.constant 1 : index
// CHECK-NEXT:            %229 = arith.addi %198, %228 : index
// CHECK-NEXT:            %230 = arith.constant 1 : index
// CHECK-NEXT:            %231 = arith.addi %199, %230 : index
// CHECK-NEXT:            memref.store %207, %186[%227, %229, %231] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}
// CHECK-NEXT: }
