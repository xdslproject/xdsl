// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
// CHECK: builtin.module {

  // The pass used to crash on external function, just regression-testing this here.
  func.func @external(!stencil.field<?xf64>) -> ()
  // CHECK:    func.func @external(memref<?xf64>) -> ()

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

// CHECK:         func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:      %3 = memref.subview %2[3, 3, 3] [64, 64, 60] [1, 1, 1] : memref<70x70x70xf64> to memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 3 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      %9 = arith.constant 1 : index
// CHECK-NEXT:      %10 = arith.constant 65 : index
// CHECK-NEXT:      %11 = arith.constant 66 : index
// CHECK-NEXT:      %12 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%4, %5, %6, %10, %11, %12, %7, %8, %9) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%13 : index, %14 : index, %15 : index):
// CHECK-NEXT:        %16 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %17 = arith.addf %0, %16 : f64
// CHECK-NEXT:        memref.store %17, %3[%13, %14, %15] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
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

// CHECK:         func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1001 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %t1_out, %t0_out = scf.for %time = %time_m to %time_M step %step iter_args(%fim1 = %f0, %fi = %f1) -> (memref<2004x2004xf32>, memref<2004x2004xf32>) {
// CHECK-NEXT:        %fi_storeview = memref.subview %fi[2, 2] [2000, 2000] [1, 1] : memref<2004x2004xf32> to memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:        %fim1_loadview = memref.subview %fim1[2, 2] [2000, 2000] [1, 1] : memref<2004x2004xf32> to memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:        %0 = arith.constant 0 : index
// CHECK-NEXT:        %1 = arith.constant 0 : index
// CHECK-NEXT:        %2 = arith.constant 1 : index
// CHECK-NEXT:        %3 = arith.constant 1 : index
// CHECK-NEXT:        %4 = arith.constant 2000 : index
// CHECK-NEXT:        %5 = arith.constant 2000 : index
// CHECK-NEXT:        "scf.parallel"(%0, %1, %4, %5, %2, %3) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^bb0(%6 : index, %7 : index):
// CHECK-NEXT:          %i = memref.load %fim1_loadview[%6, %7] : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:          memref.store %i, %fi_storeview[%6, %7] : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:          scf.reduce
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        scf.yield %fi, %fim1 : memref<2004x2004xf32>, memref<2004x2004xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %t1_out : memref<2004x2004xf32>
// CHECK-NEXT:    }

  func.func @copy_1d(%7 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
    %8 = stencil.cast %7 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
    %outc = stencil.cast %out : !stencil.field<?xf64> -> !stencil.field<[0,1024]xf64>
    %9 = stencil.load %8 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,68]xf64>
    %10 = stencil.apply(%11 = %9 : !stencil.temp<[-1,68]xf64>) -> (!stencil.temp<[0,68]xf64>) {
      %12 = stencil.access %11[-1] : !stencil.temp<[-1,68]xf64>
      stencil.return %12 : f64
    }
    stencil.store %10 to %outc (<[0], [68]>) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
    func.return
  }

// CHECK:         func.func @copy_1d(%0 : memref<?xf64>, %out : memref<?xf64>) {
// CHECK-NEXT:      %1 = "memref.cast"(%0) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:      %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
// CHECK-NEXT:      %outc_storeview = memref.subview %outc[0] [68] [1] : memref<1024xf64> to memref<68xf64, strided<[1]>>
// CHECK-NEXT:      %2 = memref.subview %1[4] [69] [1] : memref<72xf64> to memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %3 = arith.constant 0 : index
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%3, %5, %4) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb0(%6 : index):
// CHECK-NEXT:        %7 = arith.constant -1 : index
// CHECK-NEXT:        %8 = arith.addi %6, %7 : index
// CHECK-NEXT:        %9 = memref.load %2[%8] : memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %9, %outc_storeview[%6] : memref<68xf64, strided<[1]>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @copy_2d(%13 : !stencil.field<?x?xf64>) {
    %14 = stencil.cast %13 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %15 = stencil.load %14 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,64]x[0,68]xf64>
    %16 = stencil.apply(%17 = %15 : !stencil.temp<[-1,64]x[0,68]xf64>) -> (!stencil.temp<[0,64]x[0,68]xf64>) {
      %18 = stencil.access %17[-1, 0] : !stencil.temp<[-1,64]x[0,68]xf64>
      stencil.return %18 : f64
    }
    func.return
  }

// CHECK:         func.func @copy_2d(%0 : memref<?x?xf64>) {
// CHECK-NEXT:      %1 = "memref.cast"(%0) : (memref<?x?xf64>) -> memref<72x72xf64>
// CHECK-NEXT:      %2 = memref.subview %1[4, 4] [65, 68] [1, 1] : memref<72x72xf64> to memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:      %3 = arith.constant 0 : index
// CHECK-NEXT:      %4 = arith.constant 0 : index
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 64 : index
// CHECK-NEXT:      %8 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%3, %4, %7, %8, %5, %6) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb0(%9 : index, %10 : index):
// CHECK-NEXT:        %11 = arith.constant -1 : index
// CHECK-NEXT:        %12 = arith.addi %9, %11 : index
// CHECK-NEXT:        %13 = memref.load %2[%12, %10] : memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


  func.func @copy_3d(%19 : !stencil.field<?x?x?xf64>) {
    %20 = stencil.cast %19 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
    %21 = stencil.load %20 : !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64> -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
    %22 = stencil.apply(%23 = %21 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,68]xf64>) {
      %24 = stencil.access %23[-1, 0, 1] : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
      stencil.return %24 : f64
    }
    func.return
  }

// CHECK:         func.func @copy_3d(%0 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %1 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:      %2 = memref.subview %1[4, 4, 4] [65, 64, 69] [1, 1, 1] : memref<72x74x76xf64> to memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:      %3 = arith.constant 0 : index
// CHECK-NEXT:      %4 = arith.constant 0 : index
// CHECK-NEXT:      %5 = arith.constant 0 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      %9 = arith.constant 64 : index
// CHECK-NEXT:      %10 = arith.constant 64 : index
// CHECK-NEXT:      %11 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%3, %4, %5, %9, %10, %11, %6, %7, %8) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%12 : index, %13 : index, %14 : index):
// CHECK-NEXT:        %15 = arith.constant -1 : index
// CHECK-NEXT:        %16 = arith.addi %12, %15 : index
// CHECK-NEXT:        %17 = arith.constant 1 : index
// CHECK-NEXT:        %18 = arith.addi %14, %17 : index
// CHECK-NEXT:        %19 = memref.load %2[%16, %13, %18] : memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }
// CHECK:         func.func @test_funcop_lowering(%0 : memref<?x?x?xf64>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @test_funcop_lowering_dyn(%0 : memref<8x8xf64>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offsets(%27 : !stencil.field<?x?x?xf64>, %28 : !stencil.field<?x?x?xf64>, %29 : !stencil.field<?x?x?xf64>) {
    %30 = stencil.cast %27 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %31 = stencil.cast %28 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %32 = stencil.cast %29 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %33 = stencil.load %30 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %34, %35 = stencil.apply(%36 = %33 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
      %37 = stencil.access %36[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %38 = stencil.access %36[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %39 = stencil.access %36[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %40 = stencil.access %36[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %41 = stencil.access %36[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %42 = arith.addf %37, %38 : f64
      %43 = arith.addf %39, %40 : f64
      %44 = arith.addf %42, %43 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %45 = arith.mulf %41, %cst : f64
      %46 = arith.addf %45, %44 : f64
      stencil.return %46, %45 : f64, f64
    }
    stencil.store %34 to %31 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @offsets(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>, %2 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %3 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %4 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %5 = memref.subview %4[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %6 = "memref.cast"(%2) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %7 = memref.subview %3[4, 4, 4] [66, 66, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %8 = arith.constant 0 : index
// CHECK-NEXT:      %9 = arith.constant 0 : index
// CHECK-NEXT:      %10 = arith.constant 0 : index
// CHECK-NEXT:      %11 = arith.constant 1 : index
// CHECK-NEXT:      %12 = arith.constant 1 : index
// CHECK-NEXT:      %13 = arith.constant 1 : index
// CHECK-NEXT:      %14 = arith.constant 64 : index
// CHECK-NEXT:      %15 = arith.constant 64 : index
// CHECK-NEXT:      %16 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%8, %9, %10, %14, %15, %16, %11, %12, %13) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%17 : index, %18 : index, %19 : index):
// CHECK-NEXT:        %20 = arith.constant -1 : index
// CHECK-NEXT:        %21 = arith.addi %17, %20 : index
// CHECK-NEXT:        %22 = memref.load %7[%21, %18, %19] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %23 = arith.constant 1 : index
// CHECK-NEXT:        %24 = arith.addi %17, %23 : index
// CHECK-NEXT:        %25 = memref.load %7[%24, %18, %19] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %26 = arith.constant 1 : index
// CHECK-NEXT:        %27 = arith.addi %18, %26 : index
// CHECK-NEXT:        %28 = memref.load %7[%17, %27, %19] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %29 = arith.constant -1 : index
// CHECK-NEXT:        %30 = arith.addi %18, %29 : index
// CHECK-NEXT:        %31 = memref.load %7[%17, %30, %19] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %32 = memref.load %7[%17, %18, %19] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %33 = arith.addf %22, %25 : f64
// CHECK-NEXT:        %34 = arith.addf %28, %31 : f64
// CHECK-NEXT:        %35 = arith.addf %33, %34 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %36 = arith.mulf %32, %cst : f64
// CHECK-NEXT:        %37 = arith.addf %36, %35 : f64
// CHECK-NEXT:        memref.store %37, %5[%17, %18, %19] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
    stencil.external_store %dyn_field to %dyn_mem : !stencil.field<?x?x?xf64> to memref<?x?x?xf64>
    stencil.external_store %sta_field to %sta_mem : !stencil.field<[-2,62]x[0,64]x[2,66]xf64> to memref<64x64x64xf64>
    %47 = stencil.external_load %dyn_mem : memref<?x?x?xf64> -> !stencil.field<?x?x?xf64>
    %48 = stencil.external_load %sta_mem : memref<64x64x64xf64> -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
    %casted = stencil.cast %47 : !stencil.field<?x?x?xf64> -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
    func.return
  }

// CHECK:         func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : memref<?x?x?xf64>, %sta_field : memref<64x64x64xf64>) {
// CHECK-NEXT:      %casted = "memref.cast"(%dyn_mem) : (memref<?x?x?xf64>) -> memref<64x64x64xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @neg_bounds(%in : !stencil.field<[-32,32]xf64>, %out_1 : !stencil.field<[-32,32]xf64>) {
    %tin = stencil.load %in : !stencil.field<[-32,32]xf64> -> !stencil.temp<[-16,16]xf64>
    %outt = stencil.apply(%tinb = %tin : !stencil.temp<[-16,16]xf64>) -> (!stencil.temp<[-16,16]xf64>) {
      %val = stencil.access %tinb[0] : !stencil.temp<[-16,16]xf64>
      stencil.return %val : f64
    }
    stencil.store %outt to %out_1 (<[-16], [16]>) : !stencil.temp<[-16,16]xf64> to !stencil.field<[-32,32]xf64>
    func.return
  }

// CHECK:         func.func @neg_bounds(%in : memref<64xf64>, %out : memref<64xf64>) {
// CHECK-NEXT:      %out_storeview = memref.subview %out[32] [32] [1] : memref<64xf64> to memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:      %in_loadview = memref.subview %in[32] [32] [1] : memref<64xf64> to memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:      %0 = arith.constant -16 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.constant 16 : index
// CHECK-NEXT:      "scf.parallel"(%0, %2, %1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb0(%3 : index):
// CHECK-NEXT:        %val = memref.load %in_loadview[%3] : memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:        memref.store %val, %out_storeview[%3] : memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_buffer(%49 : !stencil.field<[-4,68]xf64>, %50 : !stencil.field<[-4,68]xf64>) {
    %51 = stencil.load %49 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
    %52 = stencil.apply(%53 = %51 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[1,65]xf64>) {
      %54 = stencil.access %53[-1] : !stencil.temp<[0,64]xf64>
      stencil.return %54 : f64
    }
    %55 = stencil.buffer %52 : !stencil.temp<[1,65]xf64> -> !stencil.temp<[1,65]xf64>
    %56 = stencil.apply(%57 = %55 : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
      %58 = stencil.access %57[1] : !stencil.temp<[1,65]xf64>
      stencil.return %58 : f64
    }
    stencil.store %56 to %50 (<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @stencil_buffer(%0 : memref<72xf64>, %1 : memref<72xf64>) {
// CHECK-NEXT:      %2 = memref.alloc() : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:      %3 = memref.subview %1[4] [64] [1] : memref<72xf64> to memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %4 = memref.subview %0[4] [64] [1] : memref<72xf64> to memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 65 : index
// CHECK-NEXT:      "scf.parallel"(%5, %7, %6) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb0(%8 : index):
// CHECK-NEXT:        %9 = arith.constant -1 : index
// CHECK-NEXT:        %10 = arith.addi %8, %9 : index
// CHECK-NEXT:        %11 = memref.load %4[%10] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %11, %2[%8] : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      %12 = arith.constant 0 : index
// CHECK-NEXT:      %13 = arith.constant 1 : index
// CHECK-NEXT:      %14 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%12, %14, %13) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb1(%15 : index):
// CHECK-NEXT:        %16 = arith.constant 1 : index
// CHECK-NEXT:        %17 = arith.addi %15, %16 : index
// CHECK-NEXT:        %18 = memref.load %2[%17] : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:        memref.store %18, %3[%15] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      memref.dealloc %2 : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_two_stores(%59 : !stencil.field<[-4,68]xf64>, %60 : !stencil.field<[-4,68]xf64>, %61 : !stencil.field<[-4,68]xf64>) {
    %62 = stencil.load %59 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
    %63 = stencil.apply(%64 = %62 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[1,65]xf64>) {
      %65 = stencil.access %64[-1] : !stencil.temp<[0,64]xf64>
      stencil.return %65 : f64
    }
    stencil.store %63 to %61 (<[1], [65]>) : !stencil.temp<[1,65]xf64> to !stencil.field<[-4,68]xf64>
    %66 = stencil.apply(%67 = %63 : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
      %68 = stencil.access %67[1] : !stencil.temp<[1,65]xf64>
      stencil.return %68 : f64
    }
    stencil.store %66 to %60 (<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @stencil_two_stores(%0 : memref<72xf64>, %1 : memref<72xf64>, %2 : memref<72xf64>) {
// CHECK-NEXT:      %3 = memref.subview %2[4] [64] [1] : memref<72xf64> to memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %4 = memref.subview %1[4] [64] [1] : memref<72xf64> to memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %5 = memref.subview %0[4] [64] [1] : memref<72xf64> to memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 65 : index
// CHECK-NEXT:      "scf.parallel"(%6, %8, %7) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb0(%9 : index):
// CHECK-NEXT:        %10 = arith.constant -1 : index
// CHECK-NEXT:        %11 = arith.addi %9, %10 : index
// CHECK-NEXT:        %12 = memref.load %5[%11] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %12, %3[%9] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      %13 = arith.constant 0 : index
// CHECK-NEXT:      %14 = arith.constant 1 : index
// CHECK-NEXT:      %15 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%13, %15, %14) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb1(%16 : index):
// CHECK-NEXT:        %17 = arith.constant 1 : index
// CHECK-NEXT:        %18 = arith.addi %16, %17 : index
// CHECK-NEXT:        %19 = memref.load %3[%18] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %19, %4[%16] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @apply_kernel(%69 : !stencil.field<[-2,13]x[-2,13]xf32>, %70 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_1", "u_vec", "timers"]} {
    %71 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
    %u_vec = builtin.unrealized_conversion_cast %71 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
    %72 = builtin.unrealized_conversion_cast %70 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
    "gpu.memcpy"(%71, %72) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %73 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
    %u_vec_1 = builtin.unrealized_conversion_cast %73 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
    %74 = builtin.unrealized_conversion_cast %69 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
    "gpu.memcpy"(%73, %74) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %time_m_1 = arith.constant 0 : index
    %time_M_1 = arith.constant 10 : index
    %step_1 = arith.constant 1 : index
    %75, %76 = scf.for %time_1 = %time_m_1 to %time_M_1 step %step_1 iter_args(%t0 = %u_vec_1, %t1 = %u_vec) -> (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) {
      %t0_temp = stencil.load %t0 : !stencil.field<[-2,13]x[-2,13]xf32> -> !stencil.temp<[0,11]x[0,11]xf32>
      %t1_result = stencil.apply(%t0_buff = %t0_temp : !stencil.temp<[0,11]x[0,11]xf32>) -> (!stencil.temp<[0,11]x[0,11]xf32>) {
        %77 = stencil.access %t0_buff[0, 0] : !stencil.temp<[0,11]x[0,11]xf32>
        stencil.return %77 : f32
      }
      stencil.store %t1_result to %t1 (<[0, 0], [11, 11]>) : !stencil.temp<[0,11]x[0,11]xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
      scf.yield %t1, %t0 : !stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>
    }
    func.return
  }

// CHECK:         func.func @apply_kernel(%0 : memref<15x15xf32>, %1 : memref<15x15xf32>, %timers : !llvm.ptr)  attributes {param_names = ["u_vec_1", "u_vec", "timers"]} {
// CHECK-NEXT:      %2 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec = builtin.unrealized_conversion_cast %2 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      %3 = builtin.unrealized_conversion_cast %1 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%2, %3) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %4 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec_1 = builtin.unrealized_conversion_cast %4 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      %5 = builtin.unrealized_conversion_cast %0 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%4, %5) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 10 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %6, %7 = scf.for %time = %time_m to %time_M step %step iter_args(%t0 = %u_vec_1, %t1 = %u_vec) -> (memref<15x15xf32>, memref<15x15xf32>) {
// CHECK-NEXT:        %t1_storeview = memref.subview %t1[2, 2] [11, 11] [1, 1] : memref<15x15xf32> to memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:        %t0_loadview = memref.subview %t0[2, 2] [11, 11] [1, 1] : memref<15x15xf32> to memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:        %8 = arith.constant 0 : index
// CHECK-NEXT:        %9 = arith.constant 0 : index
// CHECK-NEXT:        %10 = arith.constant 1 : index
// CHECK-NEXT:        %11 = arith.constant 1 : index
// CHECK-NEXT:        %12 = arith.constant 11 : index
// CHECK-NEXT:        %13 = arith.constant 11 : index
// CHECK-NEXT:        "scf.parallel"(%8, %9, %12, %13, %10, %11) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^bb0(%14 : index, %15 : index):
// CHECK-NEXT:          %16 = memref.load %t0_loadview[%14, %15] : memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:          memref.store %16, %t1_storeview[%14, %15] : memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:          scf.reduce
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        scf.yield %t1, %t0 : memref<15x15xf32>, memref<15x15xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_float_unrolled(%78 : f64, %79 : !stencil.field<?x?x?xf64>) {
    %80 = stencil.cast %79 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %81 = stencil.apply(%82 = %78 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
      %83 = arith.constant 1.000000e+00 : f64
      %84 = arith.constant 2.000000e+00 : f64
      %85 = arith.constant 3.000000e+00 : f64
      %86 = arith.constant 4.000000e+00 : f64
      %87 = arith.constant 5.000000e+00 : f64
      %88 = arith.constant 6.000000e+00 : f64
      %89 = arith.constant 7.000000e+00 : f64
      %90 = arith.constant 8.000000e+00 : f64
      stencil.return %83, %84, %85, %86, %87, %88, %89, %90 unroll <[2, 2, 2]> : f64, f64, f64, f64, f64, f64, f64, f64
    }
    stencil.store %81 to %80 (<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @stencil_init_float_unrolled(%0 : f64, %1 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:      %3 = memref.subview %2[3, 3, 3] [64, 64, 60] [1, 1, 1] : memref<70x70x70xf64> to memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 3 : index
// CHECK-NEXT:      %7 = arith.constant 2 : index
// CHECK-NEXT:      %8 = arith.constant 2 : index
// CHECK-NEXT:      %9 = arith.constant 2 : index
// CHECK-NEXT:      %10 = arith.constant 65 : index
// CHECK-NEXT:      %11 = arith.constant 66 : index
// CHECK-NEXT:      %12 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%4, %5, %6, %10, %11, %12, %7, %8, %9) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%13 : index, %14 : index, %15 : index):
// CHECK-NEXT:        %16 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %17 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        %18 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:        %19 = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:        %20 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:        %21 = arith.constant 6.000000e+00 : f64
// CHECK-NEXT:        %22 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:        %23 = arith.constant 8.000000e+00 : f64
// CHECK-NEXT:        memref.store %16, %3[%13, %14, %15] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %24 = arith.constant 1 : index
// CHECK-NEXT:        %25 = arith.addi %15, %24 : index
// CHECK-NEXT:        memref.store %17, %3[%13, %14, %25] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %26 = arith.constant 1 : index
// CHECK-NEXT:        %27 = arith.addi %14, %26 : index
// CHECK-NEXT:        memref.store %18, %3[%13, %27, %15] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %28 = arith.constant 1 : index
// CHECK-NEXT:        %29 = arith.addi %14, %28 : index
// CHECK-NEXT:        %30 = arith.constant 1 : index
// CHECK-NEXT:        %31 = arith.addi %15, %30 : index
// CHECK-NEXT:        memref.store %19, %3[%13, %29, %31] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %32 = arith.constant 1 : index
// CHECK-NEXT:        %33 = arith.addi %13, %32 : index
// CHECK-NEXT:        memref.store %20, %3[%33, %14, %15] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %34 = arith.constant 1 : index
// CHECK-NEXT:        %35 = arith.addi %13, %34 : index
// CHECK-NEXT:        %36 = arith.constant 1 : index
// CHECK-NEXT:        %37 = arith.addi %15, %36 : index
// CHECK-NEXT:        memref.store %21, %3[%35, %14, %37] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %38 = arith.constant 1 : index
// CHECK-NEXT:        %39 = arith.addi %13, %38 : index
// CHECK-NEXT:        %40 = arith.constant 1 : index
// CHECK-NEXT:        %41 = arith.addi %14, %40 : index
// CHECK-NEXT:        memref.store %22, %3[%39, %41, %15] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %42 = arith.constant 1 : index
// CHECK-NEXT:        %43 = arith.addi %13, %42 : index
// CHECK-NEXT:        %44 = arith.constant 1 : index
// CHECK-NEXT:        %45 = arith.addi %14, %44 : index
// CHECK-NEXT:        %46 = arith.constant 1 : index
// CHECK-NEXT:        %47 = arith.addi %15, %46 : index
// CHECK-NEXT:        memref.store %23, %3[%43, %45, %47] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_index(%91 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %92 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x = stencil.index 0 <[0, 0, 0]>
      %y = stencil.index 1 <[0, 0, 0]>
      %z = stencil.index 2 <[0, 0, 0]>
      %xy = arith.addi %x, %y : index
      %xyz = arith.addi %xy, %z : index
      stencil.return %xyz : index
    }
    stencil.store %92 to %91 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// CHECK:         func.func @stencil_init_index(%0 : memref<64x64x64xindex>) {
// CHECK-NEXT:      %1 = memref.subview %0[0, 0, 0] [64, 64, 64] [1, 1, 1] : memref<64x64x64xindex> to memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = arith.constant 0 : index
// CHECK-NEXT:      %4 = arith.constant 0 : index
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 64 : index
// CHECK-NEXT:      %9 = arith.constant 64 : index
// CHECK-NEXT:      %10 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%2, %3, %4, %8, %9, %10, %5, %6, %7) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%x : index, %y : index, %z : index):
// CHECK-NEXT:        %xy = arith.addi %x, %y : index
// CHECK-NEXT:        %xyz = arith.addi %xy, %z : index
// CHECK-NEXT:        memref.store %xyz, %1[%x, %y, %z] : memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_index_offset(%93 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %94 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x_1 = stencil.index 0 <[1, 1, 1]>
      %y_1 = stencil.index 1 <[-1, -1, -1]>
      %z_1 = stencil.index 2 <[0, 0, 0]>
      %xy_1 = arith.addi %x_1, %y_1 : index
      %xyz_1 = arith.addi %xy_1, %z_1 : index
      stencil.return %xyz_1 : index
    }
    stencil.store %94 to %93 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// CHECK:         func.func @stencil_init_index_offset(%0 : memref<64x64x64xindex>) {
// CHECK-NEXT:      %1 = memref.subview %0[0, 0, 0] [64, 64, 64] [1, 1, 1] : memref<64x64x64xindex> to memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = arith.constant 0 : index
// CHECK-NEXT:      %4 = arith.constant 0 : index
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 64 : index
// CHECK-NEXT:      %9 = arith.constant 64 : index
// CHECK-NEXT:      %10 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%2, %3, %4, %8, %9, %10, %5, %6, %7) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%11 : index, %12 : index, %z : index):
// CHECK-NEXT:        %x = arith.constant 1 : index
// CHECK-NEXT:        %x_1 = arith.addi %11, %x : index
// CHECK-NEXT:        %y = arith.constant -1 : index
// CHECK-NEXT:        %y_1 = arith.addi %12, %y : index
// CHECK-NEXT:        %xy = arith.addi %x_1, %y_1 : index
// CHECK-NEXT:        %xyz = arith.addi %xy, %z : index
// CHECK-NEXT:        memref.store %xyz, %1[%11, %12, %z] : memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @store_result_lowering(%arg0 : f64) {
    %95, %96 = stencil.apply(%arg1 = %arg0 : f64) -> (!stencil.temp<[0,7]x[0,7]x[0,7]xf64>, !stencil.temp<[0,7]x[0,7]x[0,7]xf64>) {
      %97 = stencil.store_result %arg1 : !stencil.result<f64>
      %98 = stencil.store_result %arg1 : !stencil.result<f64>
      stencil.return %97, %98 : !stencil.result<f64>, !stencil.result<f64>
    }
    %99 = stencil.buffer %96 : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> -> !stencil.temp<[0,7]x[0,7]x[0,7]xf64>
    %100 = stencil.buffer %95 : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> -> !stencil.temp<[0,7]x[0,7]x[0,7]xf64>
    func.return
  }

// CHECK:         func.func @store_result_lowering(%arg0 : f64) {
// CHECK-NEXT:      %0 = memref.alloc() : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %1 = memref.alloc() : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = arith.constant 0 : index
// CHECK-NEXT:      %4 = arith.constant 0 : index
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 7 : index
// CHECK-NEXT:      %9 = arith.constant 7 : index
// CHECK-NEXT:      %10 = arith.constant 7 : index
// CHECK-NEXT:      "scf.parallel"(%2, %3, %4, %8, %9, %10, %5, %6, %7) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%11 : index, %12 : index, %13 : index):
// CHECK-NEXT:        memref.store %arg0, %1[%11, %12, %13] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:        memref.store %arg0, %0[%11, %12, %13] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      memref.dealloc %0 : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      memref.dealloc %1 : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @if_lowering(%arg0_1 : f64, %b0 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>, %b1 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>)  attributes {"stencil.program"} {
    %101, %102 = stencil.apply(%arg1_1 = %arg0_1 : f64) -> (!stencil.temp<[0,7]x[0,7]x[0,7]xf64>, !stencil.temp<[0,7]x[0,7]x[0,7]xf64>) {
      %true = "test.op"() : () -> i1
      %103, %104 = scf.if %true -> (!stencil.result<f64>, f64) {
        %105 = stencil.store_result %arg1_1 : !stencil.result<f64>
        scf.yield %105, %arg1_1 : !stencil.result<f64>, f64
      } else {
        %106 = stencil.store_result  : !stencil.result<f64>
        scf.yield %106, %arg1_1 : !stencil.result<f64>, f64
      }
      %107 = stencil.store_result %104 : !stencil.result<f64>
      stencil.return %103, %107 : !stencil.result<f64>, !stencil.result<f64>
    }
    stencil.store %101 to %b0 (<[0, 0, 0], [7, 7, 7]>) : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> to !stencil.field<[0,7]x[0,7]x[0,7]xf64>
    stencil.store %102 to %b1 (<[0, 0, 0], [7, 7, 7]>) : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> to !stencil.field<[0,7]x[0,7]x[0,7]xf64>
    func.return
  }

// CHECK:         func.func @if_lowering(%arg0 : f64, %b0 : memref<7x7x7xf64>, %b1 : memref<7x7x7xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %b0_storeview = memref.subview %b0[0, 0, 0] [7, 7, 7] [1, 1, 1] : memref<7x7x7xf64> to memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %b1_storeview = memref.subview %b1[0, 0, 0] [7, 7, 7] [1, 1, 1] : memref<7x7x7xf64> to memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 0 : index
// CHECK-NEXT:      %2 = arith.constant 0 : index
// CHECK-NEXT:      %3 = arith.constant 1 : index
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 7 : index
// CHECK-NEXT:      %7 = arith.constant 7 : index
// CHECK-NEXT:      %8 = arith.constant 7 : index
// CHECK-NEXT:      "scf.parallel"(%0, %1, %2, %6, %7, %8, %3, %4, %5) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%9 : index, %10 : index, %11 : index):
// CHECK-NEXT:        %true = "test.op"() : () -> i1
// CHECK-NEXT:        %12, %13 = scf.if %true -> (f64, f64) {
// CHECK-NEXT:          memref.store %arg0, %b0_storeview[%9, %10, %11] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:          scf.yield %arg0, %arg0 : f64, f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %14 = builtin.unrealized_conversion_cast to f64
// CHECK-NEXT:          scf.yield %14, %arg0 : f64, f64
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %13, %b1_storeview[%9, %10, %11] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @combine(%108 : !stencil.field<?x?xf64>) {
    %109 = stencil.cast %108 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
    %110 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
      %111 = arith.constant 1.000000e+00 : f64
      stencil.return %111 : f64
    }
    %112 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
      %113 = arith.constant 2.000000e+00 : f64
      stencil.return %113 : f64
    }
    %114 = stencil.combine 0 at 33 lower = (%110 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%112 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
    stencil.store %114 to %109 (<[1, 2], [65, 66]>) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @combine(%0 : memref<?x?xf64>) {
// CHECK-NEXT:      %1 = "memref.cast"(%0) : (memref<?x?xf64>) -> memref<70x70xf64>
// CHECK-NEXT:      %2 = memref.subview %1[3, 3] [64, 64] [1, 1] : memref<70x70xf64> to memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:      %3 = arith.constant 1 : index
// CHECK-NEXT:      %4 = arith.constant 2 : index
// CHECK-NEXT:      %5 = arith.constant 1 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 33 : index
// CHECK-NEXT:      %8 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%3, %4, %7, %8, %5, %6) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb0(%9 : index, %10 : index):
// CHECK-NEXT:        %11 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        memref.store %11, %2[%9, %10] : memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %12 = arith.constant 33 : index
// CHECK-NEXT:      %13 = arith.constant 2 : index
// CHECK-NEXT:      %14 = arith.constant 1 : index
// CHECK-NEXT:      %15 = arith.constant 1 : index
// CHECK-NEXT:      %16 = arith.constant 65 : index
// CHECK-NEXT:      %17 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%12, %13, %16, %17, %14, %15) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb1(%18 : index, %19 : index):
// CHECK-NEXT:        %20 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        memref.store %20, %2[%18, %19] : memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @buffered_combine(%115 : !stencil.field<?x?xf64>) {
    %116 = stencil.cast %115 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
    %117 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
      %118 = arith.constant 1.000000e+00 : f64
      stencil.return %118 : f64
    }
    %119 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
      %120 = arith.constant 2.000000e+00 : f64
      stencil.return %120 : f64
    }
    %121 = stencil.combine 0 at 33 lower = (%117 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%119 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
    %122 = stencil.buffer %121 : !stencil.temp<[1,65]x[2,66]xf64> -> !stencil.temp<[1,65]x[2,66]xf64>
    %123 = stencil.apply(%124 = %122 : !stencil.temp<[1,65]x[2,66]xf64>) -> (!stencil.temp<[1,65]x[2,66]xf64>) {
      %125 = arith.constant 1.000000e+00 : f64
      %126 = stencil.access %124[0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
      %127 = arith.addf %125, %126 : f64
      stencil.return %127 : f64
    }
    stencil.store %123 to %116 (<[1, 2], [65, 66]>) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @buffered_combine(%0 : memref<?x?xf64>) {
// CHECK-NEXT:      %1 = memref.alloc() : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:      %2 = "memref.cast"(%0) : (memref<?x?xf64>) -> memref<70x70xf64>
// CHECK-NEXT:      %3 = memref.subview %2[3, 3] [64, 64] [1, 1] : memref<70x70xf64> to memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:      %4 = arith.constant 1 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 1 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 33 : index
// CHECK-NEXT:      %9 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%4, %5, %8, %9, %6, %7) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb0(%10 : index, %11 : index):
// CHECK-NEXT:        %12 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        memref.store %12, %1[%10, %11] : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %13 = arith.constant 33 : index
// CHECK-NEXT:      %14 = arith.constant 2 : index
// CHECK-NEXT:      %15 = arith.constant 1 : index
// CHECK-NEXT:      %16 = arith.constant 1 : index
// CHECK-NEXT:      %17 = arith.constant 65 : index
// CHECK-NEXT:      %18 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%13, %14, %17, %18, %15, %16) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb1(%19 : index, %20 : index):
// CHECK-NEXT:        %21 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        memref.store %21, %1[%19, %20] : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %22 = arith.constant 1 : index
// CHECK-NEXT:      %23 = arith.constant 2 : index
// CHECK-NEXT:      %24 = arith.constant 1 : index
// CHECK-NEXT:      %25 = arith.constant 1 : index
// CHECK-NEXT:      %26 = arith.constant 65 : index
// CHECK-NEXT:      %27 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%22, %23, %26, %27, %24, %25) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb2(%28 : index, %29 : index):
// CHECK-NEXT:        %30 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %31 = memref.load %1[%28, %29] : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:        %32 = arith.addf %30, %31 : f64
// CHECK-NEXT:        memref.store %32, %3[%28, %29] : memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      memref.dealloc %1 : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offset_mapping(%128 : !stencil.field<[0,8]xf64>, %129 : !stencil.field<[0,8]xf64>, %130 : !stencil.field<[0,8]x[0,8]xf64>) {
    %131 = stencil.load %128 : !stencil.field<[0,8]xf64> -> !stencil.temp<[0,8]xf64>
    %132 = stencil.load %129 : !stencil.field<[0,8]xf64> -> !stencil.temp<[0,8]xf64>
    %133 = stencil.apply(%134 = %131 : !stencil.temp<[0,8]xf64>, %135 = %132 : !stencil.temp<[0,8]xf64>) -> (!stencil.temp<[0,8]x[0,8]xf64>) {
      %136 = stencil.access %134[0, _] : !stencil.temp<[0,8]xf64>
      %137 = stencil.access %135[_, 0] : !stencil.temp<[0,8]xf64>
      %138 = arith.mulf %136, %137 : f64
      stencil.return %138 : f64
    }
    stencil.store %133 to %130 (<[0], [8]>) : !stencil.temp<[0,8]x[0,8]xf64> to !stencil.field<[0,8]x[0,8]xf64>
    func.return
  }

// CHECK:         func.func @offset_mapping(%0 : memref<8xf64>, %1 : memref<8xf64>, %2 : memref<8x8xf64>) {
// CHECK-NEXT:      %3 = memref.subview %2[0, 0] [8, 8] [1, 1] : memref<8x8xf64> to memref<8x8xf64, strided<[8, 1]>>
// CHECK-NEXT:      %4 = memref.subview %0[0] [8] [1] : memref<8xf64> to memref<8xf64, strided<[1]>>
// CHECK-NEXT:      %5 = memref.subview %1[0] [8] [1] : memref<8xf64> to memref<8xf64, strided<[1]>>
// CHECK-NEXT:      %6 = arith.constant 0 : index
// CHECK-NEXT:      %7 = arith.constant 0 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      %9 = arith.constant 1 : index
// CHECK-NEXT:      %10 = arith.constant 8 : index
// CHECK-NEXT:      %11 = arith.constant 8 : index
// CHECK-NEXT:      "scf.parallel"(%6, %7, %10, %11, %8, %9) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^bb0(%12 : index, %13 : index):
// CHECK-NEXT:        %14 = memref.load %4[%12] : memref<8xf64, strided<[1]>>
// CHECK-NEXT:        %15 = memref.load %5[%13] : memref<8xf64, strided<[1]>>
// CHECK-NEXT:        %16 = arith.mulf %14, %15 : f64
// CHECK-NEXT:        memref.store %16, %3[%12, %13] : memref<8x8xf64, strided<[8, 1]>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_copy_bufferized(%0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
    stencil.apply(%6 = %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
      %7 = stencil.access %6[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
      %8 = stencil.store_result %7 : !stencil.result<f64>
      stencil.return %8 : !stencil.result<f64>
    } to <[0, 0, 0], [64, 64, 64]>
    func.return
  }

// CHECK:    func.func @stencil_copy_bufferized(%0 : memref<72x72x72xf64>, %1 : memref<72x72x72xf64>) {
// CHECK-NEXT:      %2 = memref.subview %1[4, 4, 4] [72, 72, 72] [1, 1, 1] : memref<72x72x72xf64> to memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %3 = memref.subview %0[4, 4, 4] [72, 72, 72] [1, 1, 1] : memref<72x72x72xf64> to memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %4 = arith.constant 0 : index
// CHECK-NEXT:      %5 = arith.constant 0 : index
// CHECK-NEXT:      %6 = arith.constant 0 : index
// CHECK-NEXT:      %7 = arith.constant 1 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      %9 = arith.constant 1 : index
// CHECK-NEXT:      %10 = arith.constant 64 : index
// CHECK-NEXT:      %11 = arith.constant 64 : index
// CHECK-NEXT:      %12 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%4, %5, %6, %10, %11, %12, %7, %8, %9) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^bb0(%13 : index, %14 : index, %15 : index):
// CHECK-NEXT:        %16 = memref.load %3[%13, %14, %15] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        memref.store %16, %2[%13, %14, %15] : memref<72x72x72xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}
// CHECK-NEXT: }

// -----

  func.func @stencil_double_store(%93 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %94 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x_1 = stencil.index 0 <[1, 1, 1]>
      %y_1 = stencil.index 1 <[-1, -1, -1]>
      %z_1 = stencil.index 2 <[0, 0, 0]>
      %xy_1 = arith.addi %x_1, %y_1 : index
      %xyz_1 = arith.addi %xy_1, %z_1 : index
      stencil.return %xyz_1 : index
    }
    stencil.store %94 to %93 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    %95 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x_1 = stencil.index 0 <[1, 1, 1]>
      %y_1 = stencil.index 1 <[-1, -1, -1]>
      %z_1 = stencil.index 2 <[0, 0, 0]>
      %xy_1 = arith.addi %x_1, %y_1 : index
      %xyz_1 = arith.addi %xy_1, %z_1 : index
      stencil.return %xyz_1 : index
    }
    stencil.store %95 to %93 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// CHECK:           "stencil.store"(%2, %0) {bounds = #stencil.bounds<[0, 0, 0], [64, 64, 64]>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>, !stencil.field<[0,64]x[0,64]x[0,64]xindex>) -> ()
// CHECK-NEXT:      ^^^^^^^^^^^^^^^-------------------------------------------------------------------------------------------------------------------------
// CHECK-NEXT:      | Error while applying pattern: Cannot lower directly if storing to the same field multiple times! Try running `stencil-bufferize` before.
// CHECK-NEXT:      ----------------------------------------------------------------------------------------------------------------------------------------

// -----

  func.func @stencil_load_store(%93 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %94 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x_1 = stencil.index 0 <[1, 1, 1]>
      %y_1 = stencil.index 1 <[-1, -1, -1]>
      %z_1 = stencil.index 2 <[0, 0, 0]>
      %xy_1 = arith.addi %x_1, %y_1 : index
      %xyz_1 = arith.addi %xy_1, %z_1 : index
      stencil.return %xyz_1 : index
    }
    stencil.store %94 to %93 (<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    %95 = stencil.load %93 : !stencil.field<[0,64]x[0,64]x[0,64]xindex> -> !stencil.temp<[0,64]x[0,64]x[0,64]xindex>
    "test.op"(%95) : (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) -> ()
    func.return
  }

// CHECK:           %2 = "stencil.load"(%0) : (!stencil.field<[0,64]x[0,64]x[0,64]xindex>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xindex>
// CHECK-NEXT:      ^^^^^^^^^^^^^^^^^^^-----------------------------------------------------------------------------------------------------------------
// CHECK-NEXT:      | Error while applying pattern: Cannot lower directly if loading and storing the same field! Try running `stencil-bufferize` before.
// CHECK-NEXT:      ------------------------------------------------------------------------------------------------------------------------------------
