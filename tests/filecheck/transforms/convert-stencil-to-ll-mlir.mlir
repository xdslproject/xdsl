// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir | filecheck %s

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
    stencil.store %3 to %2 ([1, 2, 3] : [65, 66, 63]) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
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
// CHECK-NEXT:      %10 = arith.constant 65 : index
// CHECK-NEXT:      %11 = arith.constant 66 : index
// CHECK-NEXT:      %12 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%4, %5, %6, %10, %11, %12, %7, %8, %9) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^0(%13 : index, %14 : index, %15 : index):
// CHECK-NEXT:        %16 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %17 = arith.addf %0, %16 : f64
// CHECK-NEXT:        memref.store %17, %3[%13, %14, %15] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        scf.yield
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
        %i = stencil.access %tim1_b [0, 0] : !stencil.temp<[0,2000]x[0,2000]xf32>
        stencil.return %i : f32
      }
      stencil.store %ti to %fi ([0, 0] : [2000, 2000]) : !stencil.temp<[0,2000]x[0,2000]xf32> to !stencil.field<[-2,2002]x[-2,2002]xf32>
      scf.yield %fi, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>
    }
    func.return %t1_out : !stencil.field<[-2,2002]x[-2,2002]xf32>
  }

// CHECK:         func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1001 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %t1_out, %t0_out = scf.for %time = %time_m to %time_M step %step iter_args(%fim1 = %f0, %fi = %f1) -> (memref<2004x2004xf32>, memref<2004x2004xf32>) {
// CHECK-NEXT:        %fi_storeview = "memref.subview"(%fi) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:        %fim1_loadview = "memref.subview"(%fim1) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:        %18 = arith.constant 0 : index
// CHECK-NEXT:        %19 = arith.constant 0 : index
// CHECK-NEXT:        %20 = arith.constant 1 : index
// CHECK-NEXT:        %21 = arith.constant 1 : index
// CHECK-NEXT:        %22 = arith.constant 2000 : index
// CHECK-NEXT:        %23 = arith.constant 2000 : index
// CHECK-NEXT:        "scf.parallel"(%18, %19, %22, %23, %20, %21) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^1(%24 : index, %25 : index):
// CHECK-NEXT:          %i = memref.load %fim1_loadview[%24, %25] : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:          memref.store %i, %fi_storeview[%24, %25] : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:          scf.yield
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
      %12 = stencil.access %11 [-1] : !stencil.temp<[-1,68]xf64>
      stencil.return %12 : f64
    }
    stencil.store %10 to %outc ([0] : [68]) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
    func.return
  }

// CHECK:         func.func @copy_1d(%26 : memref<?xf64>, %out : memref<?xf64>) {
// CHECK-NEXT:      %27 = "memref.cast"(%26) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:      %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
// CHECK-NEXT:      %outc_storeview = "memref.subview"(%outc) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
// CHECK-NEXT:      %28 = "memref.subview"(%27) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %29 = arith.constant 0 : index
// CHECK-NEXT:      %30 = arith.constant 1 : index
// CHECK-NEXT:      %31 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%29, %31, %30) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^2(%32 : index):
// CHECK-NEXT:        %33 = arith.constant -1 : index
// CHECK-NEXT:        %34 = arith.addi %32, %33 : index
// CHECK-NEXT:        %35 = memref.load %28[%34] : memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %35, %outc_storeview[%32] : memref<68xf64, strided<[1]>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @copy_2d(%13 : !stencil.field<?x?xf64>) {
    %14 = stencil.cast %13 : !stencil.field<?x?xf64> -> !stencil.field<[-4,68]x[-4,68]xf64>
    %15 = stencil.load %14 : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,64]x[0,68]xf64>
    %16 = stencil.apply(%17 = %15 : !stencil.temp<[-1,64]x[0,68]xf64>) -> (!stencil.temp<[0,64]x[0,68]xf64>) {
      %18 = stencil.access %17 [-1, 0] : !stencil.temp<[-1,64]x[0,68]xf64>
      stencil.return %18 : f64
    }
    func.return
  }

// CHECK:         func.func @copy_2d(%36 : memref<?x?xf64>) {
// CHECK-NEXT:      %37 = "memref.cast"(%36) : (memref<?x?xf64>) -> memref<72x72xf64>
// CHECK-NEXT:      %38 = "memref.subview"(%37) <{"static_offsets" = array<i64: 4, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:      %39 = arith.constant 0 : index
// CHECK-NEXT:      %40 = arith.constant 0 : index
// CHECK-NEXT:      %41 = arith.constant 1 : index
// CHECK-NEXT:      %42 = arith.constant 1 : index
// CHECK-NEXT:      %43 = arith.constant 64 : index
// CHECK-NEXT:      %44 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%39, %40, %43, %44, %41, %42) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^3(%45 : index, %46 : index):
// CHECK-NEXT:        %47 = arith.constant -1 : index
// CHECK-NEXT:        %48 = arith.addi %45, %47 : index
// CHECK-NEXT:        %49 = memref.load %38[%48, %46] : memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


  func.func @copy_3d(%19 : !stencil.field<?x?x?xf64>) {
    %20 = stencil.cast %19 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
    %21 = stencil.load %20 : !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64> -> !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
    %22 = stencil.apply(%23 = %21 : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,68]xf64>) {
      %24 = stencil.access %23 [-1, 0, 1] : !stencil.temp<[-1,64]x[0,64]x[0,69]xf64>
      stencil.return %24 : f64
    }
    func.return
  }

// CHECK:         func.func @copy_3d(%50 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %51 = "memref.cast"(%50) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:      %52 = "memref.subview"(%51) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:      %53 = arith.constant 0 : index
// CHECK-NEXT:      %54 = arith.constant 0 : index
// CHECK-NEXT:      %55 = arith.constant 0 : index
// CHECK-NEXT:      %56 = arith.constant 1 : index
// CHECK-NEXT:      %57 = arith.constant 1 : index
// CHECK-NEXT:      %58 = arith.constant 1 : index
// CHECK-NEXT:      %59 = arith.constant 64 : index
// CHECK-NEXT:      %60 = arith.constant 64 : index
// CHECK-NEXT:      %61 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%53, %54, %55, %59, %60, %61, %56, %57, %58) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^4(%62 : index, %63 : index, %64 : index):
// CHECK-NEXT:        %65 = arith.constant -1 : index
// CHECK-NEXT:        %66 = arith.addi %62, %65 : index
// CHECK-NEXT:        %67 = arith.constant 1 : index
// CHECK-NEXT:        %68 = arith.addi %64, %67 : index
// CHECK-NEXT:        %69 = memref.load %52[%66, %63, %68] : memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }
// CHECK:         func.func @test_funcop_lowering(%70 : memref<?x?x?xf64>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @test_funcop_lowering_dyn(%71 : memref<8x8xf64>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offsets(%27 : !stencil.field<?x?x?xf64>, %28 : !stencil.field<?x?x?xf64>, %29 : !stencil.field<?x?x?xf64>) {
    %30 = stencil.cast %27 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %31 = stencil.cast %28 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %32 = stencil.cast %29 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %33 = stencil.load %30 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %34, %35 = stencil.apply(%36 = %33 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
      %37 = stencil.access %36 [-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %38 = stencil.access %36 [1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %39 = stencil.access %36 [0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %40 = stencil.access %36 [0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %41 = stencil.access %36 [0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %42 = arith.addf %37, %38 : f64
      %43 = arith.addf %39, %40 : f64
      %44 = arith.addf %42, %43 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %45 = arith.mulf %41, %cst : f64
      %46 = arith.addf %45, %44 : f64
      stencil.return %46, %45 : f64, f64
    }
    stencil.store %34 to %31 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @offsets(%72 : memref<?x?x?xf64>, %73 : memref<?x?x?xf64>, %74 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %75 = "memref.cast"(%72) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %76 = "memref.cast"(%73) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %77 = "memref.subview"(%76) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %78 = "memref.cast"(%74) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %79 = "memref.subview"(%75) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %80 = arith.constant 0 : index
// CHECK-NEXT:      %81 = arith.constant 0 : index
// CHECK-NEXT:      %82 = arith.constant 0 : index
// CHECK-NEXT:      %83 = arith.constant 1 : index
// CHECK-NEXT:      %84 = arith.constant 1 : index
// CHECK-NEXT:      %85 = arith.constant 1 : index
// CHECK-NEXT:      %86 = arith.constant 64 : index
// CHECK-NEXT:      %87 = arith.constant 64 : index
// CHECK-NEXT:      %88 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%80, %81, %82, %86, %87, %88, %83, %84, %85) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^5(%89 : index, %90 : index, %91 : index):
// CHECK-NEXT:        %92 = arith.constant -1 : index
// CHECK-NEXT:        %93 = arith.addi %89, %92 : index
// CHECK-NEXT:        %94 = memref.load %79[%93, %90, %91] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %95 = arith.constant 1 : index
// CHECK-NEXT:        %96 = arith.addi %89, %95 : index
// CHECK-NEXT:        %97 = memref.load %79[%96, %90, %91] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %98 = arith.constant 1 : index
// CHECK-NEXT:        %99 = arith.addi %90, %98 : index
// CHECK-NEXT:        %100 = memref.load %79[%89, %99, %91] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %101 = arith.constant -1 : index
// CHECK-NEXT:        %102 = arith.addi %90, %101 : index
// CHECK-NEXT:        %103 = memref.load %79[%89, %102, %91] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %104 = memref.load %79[%89, %90, %91] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        %105 = arith.addf %94, %97 : f64
// CHECK-NEXT:        %106 = arith.addf %100, %103 : f64
// CHECK-NEXT:        %107 = arith.addf %105, %106 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %108 = arith.mulf %104, %cst : f64
// CHECK-NEXT:        %109 = arith.addf %108, %107 : f64
// CHECK-NEXT:        memref.store %109, %77[%89, %90, %91] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
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

  func.func @neg_bounds(%in : !stencil.field<[-32,32]xf64>, %out_1 : !stencil.field<[-32,32]xf64>) {
    %tin = stencil.load %in : !stencil.field<[-32,32]xf64> -> !stencil.temp<[-16,16]xf64>
    %outt = stencil.apply(%tinb = %tin : !stencil.temp<[-16,16]xf64>) -> (!stencil.temp<[-16,16]xf64>) {
      %val = stencil.access %tinb [0] : !stencil.temp<[-16,16]xf64>
      stencil.return %val : f64
    }
    stencil.store %outt to %out_1 ([-16] : [16]) : !stencil.temp<[-16,16]xf64> to !stencil.field<[-32,32]xf64>
    func.return
  }

// CHECK:         func.func @neg_bounds(%in : memref<64xf64>, %out_1 : memref<64xf64>) {
// CHECK-NEXT:      %out_1_storeview = "memref.subview"(%out_1) <{"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:      %in_loadview = "memref.subview"(%in) <{"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:      %110 = arith.constant -16 : index
// CHECK-NEXT:      %111 = arith.constant 1 : index
// CHECK-NEXT:      %112 = arith.constant 16 : index
// CHECK-NEXT:      "scf.parallel"(%110, %112, %111) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^6(%113 : index):
// CHECK-NEXT:        %val = memref.load %in_loadview[%113] : memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:        memref.store %val, %out_1_storeview[%113] : memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_buffer(%49 : !stencil.field<[-4,68]xf64>, %50 : !stencil.field<[-4,68]xf64>) {
    %51 = stencil.load %49 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
    %52 = stencil.apply(%53 = %51 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[1,65]xf64>) {
      %54 = stencil.access %53 [-1] : !stencil.temp<[0,64]xf64>
      stencil.return %54 : f64
    }
    %55 = stencil.buffer %52 : !stencil.temp<[1,65]xf64>
    %56 = stencil.apply(%57 = %55 : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
      %58 = stencil.access %57 [1] : !stencil.temp<[1,65]xf64>
      stencil.return %58 : f64
    }
    stencil.store %56 to %50 ([0] : [64]) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK-NEXT:    func.func @stencil_buffer(%114 : memref<72xf64>, %115 : memref<72xf64>) {
// CHECK-NEXT:      %116 = memref.alloc() : memref<64xf64>
// CHECK-NEXT:      %117 = "memref.subview"(%116) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64xf64>) -> memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:      %118 = "memref.subview"(%115) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %119 = "memref.subview"(%114) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %120 = arith.constant 1 : index
// CHECK-NEXT:      %121 = arith.constant 1 : index
// CHECK-NEXT:      %122 = arith.constant 65 : index
// CHECK-NEXT:      "scf.parallel"(%120, %122, %121) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^7(%123 : index):
// CHECK-NEXT:        %124 = arith.constant -1 : index
// CHECK-NEXT:        %125 = arith.addi %123, %124 : index
// CHECK-NEXT:        %126 = memref.load %119[%125] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %126, %117[%123] : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      %127 = arith.constant 0 : index
// CHECK-NEXT:      %128 = arith.constant 1 : index
// CHECK-NEXT:      %129 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%127, %129, %128) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^8(%130 : index):
// CHECK-NEXT:        %131 = arith.constant 1 : index
// CHECK-NEXT:        %132 = arith.addi %130, %131 : index
// CHECK-NEXT:        %133 = memref.load %117[%132] : memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:        memref.store %133, %118[%130] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      memref.dealloc %116 : memref<64xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_two_stores(%59 : !stencil.field<[-4,68]xf64>, %60 : !stencil.field<[-4,68]xf64>, %61 : !stencil.field<[-4,68]xf64>) {
    %62 = stencil.load %59 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
    %63 = stencil.apply(%64 = %62 : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[1,65]xf64>) {
      %65 = stencil.access %64 [-1] : !stencil.temp<[0,64]xf64>
      stencil.return %65 : f64
    }
    stencil.store %63 to %61 ([1] : [65]) : !stencil.temp<[1,65]xf64> to !stencil.field<[-4,68]xf64>
    %66 = stencil.apply(%67 = %63 : !stencil.temp<[1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
      %68 = stencil.access %67 [1] : !stencil.temp<[1,65]xf64>
      stencil.return %68 : f64
    }
    stencil.store %66 to %60 ([0] : [64]) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @stencil_two_stores(%134 : memref<72xf64>, %135 : memref<72xf64>, %136 : memref<72xf64>) {
// CHECK-NEXT:      %137 = "memref.subview"(%135) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %138 = "memref.subview"(%136) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %139 = "memref.subview"(%134) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %140 = arith.constant 1 : index
// CHECK-NEXT:      %141 = arith.constant 1 : index
// CHECK-NEXT:      %142 = arith.constant 65 : index
// CHECK-NEXT:      "scf.parallel"(%140, %142, %141) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^9(%143 : index):
// CHECK-NEXT:        %144 = arith.constant -1 : index
// CHECK-NEXT:        %145 = arith.addi %143, %144 : index
// CHECK-NEXT:        %146 = memref.load %139[%145] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %146, %138[%143] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      %147 = arith.constant 0 : index
// CHECK-NEXT:      %148 = arith.constant 1 : index
// CHECK-NEXT:      %149 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%147, %149, %148) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^10(%150 : index):
// CHECK-NEXT:        %151 = arith.constant 1 : index
// CHECK-NEXT:        %152 = arith.addi %150, %151 : index
// CHECK-NEXT:        %153 = memref.load %138[%152] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        memref.store %153, %137[%150] : memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @apply_kernel(%69 : !stencil.field<[-2,13]x[-2,13]xf32>, %70 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
    %71 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
    %u_vec_1 = builtin.unrealized_conversion_cast %71 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
    %72 = builtin.unrealized_conversion_cast %70 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
    "gpu.memcpy"(%71, %72) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %73 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
    %u_vec_0 = builtin.unrealized_conversion_cast %73 : memref<15x15xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
    %74 = builtin.unrealized_conversion_cast %69 : !stencil.field<[-2,13]x[-2,13]xf32> to memref<15x15xf32>
    "gpu.memcpy"(%73, %74) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %time_m_1 = arith.constant 0 : index
    %time_M_1 = arith.constant 10 : index
    %step_1 = arith.constant 1 : index
    %75, %76 = scf.for %time_1 = %time_m_1 to %time_M_1 step %step_1 iter_args(%t0 = %u_vec_0, %t1 = %u_vec_1) -> (!stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>) {
      %t0_temp = stencil.load %t0 : !stencil.field<[-2,13]x[-2,13]xf32> -> !stencil.temp<[0,11]x[0,11]xf32>
      %t1_result = stencil.apply(%t0_buff = %t0_temp : !stencil.temp<[0,11]x[0,11]xf32>) -> (!stencil.temp<[0,11]x[0,11]xf32>) {
        %77 = stencil.access %t0_buff [0, 0] : !stencil.temp<[0,11]x[0,11]xf32>
        stencil.return %77 : f32
      }
      stencil.store %t1_result to %t1 ([0, 0] : [11, 11]) : !stencil.temp<[0,11]x[0,11]xf32> to !stencil.field<[-2,13]x[-2,13]xf32>
      scf.yield %t1, %t0 : !stencil.field<[-2,13]x[-2,13]xf32>, !stencil.field<[-2,13]x[-2,13]xf32>
    }
    func.return
  }

// CHECK:         func.func @apply_kernel(%154 : memref<15x15xf32>, %155 : memref<15x15xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
// CHECK-NEXT:      %156 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec_1 = builtin.unrealized_conversion_cast %156 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      %157 = builtin.unrealized_conversion_cast %155 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%156, %157) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %158 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
// CHECK-NEXT:      %u_vec_0 = builtin.unrealized_conversion_cast %158 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      %159 = builtin.unrealized_conversion_cast %154 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:      "gpu.memcpy"(%158, %159) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:      %time_m_1 = arith.constant 0 : index
// CHECK-NEXT:      %time_M_1 = arith.constant 10 : index
// CHECK-NEXT:      %step_1 = arith.constant 1 : index
// CHECK-NEXT:      %160, %161 = scf.for %time_1 = %time_m_1 to %time_M_1 step %step_1 iter_args(%t0 = %u_vec_0, %t1 = %u_vec_1) -> (memref<15x15xf32>, memref<15x15xf32>) {
// CHECK-NEXT:        %t1_storeview = "memref.subview"(%t1) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<15x15xf32>) -> memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:        %t0_loadview = "memref.subview"(%t0) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<15x15xf32>) -> memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:        %162 = arith.constant 0 : index
// CHECK-NEXT:        %163 = arith.constant 0 : index
// CHECK-NEXT:        %164 = arith.constant 1 : index
// CHECK-NEXT:        %165 = arith.constant 1 : index
// CHECK-NEXT:        %166 = arith.constant 11 : index
// CHECK-NEXT:        %167 = arith.constant 11 : index
// CHECK-NEXT:        "scf.parallel"(%162, %163, %166, %167, %164, %165) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^11(%168 : index, %169 : index):
// CHECK-NEXT:          %170 = memref.load %t0_loadview[%168, %169] : memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:          memref.store %170, %t1_storeview[%168, %169] : memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:          scf.yield
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
      stencil.return %83, %84, %85, %86, %87, %88, %89, %90 unroll [2, 2, 2] : f64, f64, f64, f64, f64, f64, f64, f64
    }
    stencil.store %81 to %80 ([1, 2, 3] : [65, 66, 63]) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @stencil_init_float_unrolled(%171 : f64, %172 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %173 = "memref.cast"(%172) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:      %174 = "memref.subview"(%173) <{"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:      %175 = arith.constant 1 : index
// CHECK-NEXT:      %176 = arith.constant 2 : index
// CHECK-NEXT:      %177 = arith.constant 3 : index
// CHECK-NEXT:      %178 = arith.constant 2 : index
// CHECK-NEXT:      %179 = arith.constant 2 : index
// CHECK-NEXT:      %180 = arith.constant 2 : index
// CHECK-NEXT:      %181 = arith.constant 65 : index
// CHECK-NEXT:      %182 = arith.constant 66 : index
// CHECK-NEXT:      %183 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%175, %176, %177, %181, %182, %183, %178, %179, %180) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^12(%184 : index, %185 : index, %186 : index):
// CHECK-NEXT:        %187 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %188 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        %189 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:        %190 = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:        %191 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:        %192 = arith.constant 6.000000e+00 : f64
// CHECK-NEXT:        %193 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:        %194 = arith.constant 8.000000e+00 : f64
// CHECK-NEXT:        memref.store %187, %174[%184, %185, %186] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %195 = arith.constant 1 : index
// CHECK-NEXT:        %196 = arith.addi %186, %195 : index
// CHECK-NEXT:        memref.store %188, %174[%184, %185, %196] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %197 = arith.constant 1 : index
// CHECK-NEXT:        %198 = arith.addi %185, %197 : index
// CHECK-NEXT:        memref.store %189, %174[%184, %198, %186] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %199 = arith.constant 1 : index
// CHECK-NEXT:        %200 = arith.addi %185, %199 : index
// CHECK-NEXT:        %201 = arith.constant 1 : index
// CHECK-NEXT:        %202 = arith.addi %186, %201 : index
// CHECK-NEXT:        memref.store %190, %174[%184, %200, %202] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %203 = arith.constant 1 : index
// CHECK-NEXT:        %204 = arith.addi %184, %203 : index
// CHECK-NEXT:        memref.store %191, %174[%204, %185, %186] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %205 = arith.constant 1 : index
// CHECK-NEXT:        %206 = arith.addi %184, %205 : index
// CHECK-NEXT:        %207 = arith.constant 1 : index
// CHECK-NEXT:        %208 = arith.addi %186, %207 : index
// CHECK-NEXT:        memref.store %192, %174[%206, %185, %208] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %209 = arith.constant 1 : index
// CHECK-NEXT:        %210 = arith.addi %184, %209 : index
// CHECK-NEXT:        %211 = arith.constant 1 : index
// CHECK-NEXT:        %212 = arith.addi %185, %211 : index
// CHECK-NEXT:        memref.store %193, %174[%210, %212, %186] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        %213 = arith.constant 1 : index
// CHECK-NEXT:        %214 = arith.addi %184, %213 : index
// CHECK-NEXT:        %215 = arith.constant 1 : index
// CHECK-NEXT:        %216 = arith.addi %185, %215 : index
// CHECK-NEXT:        %217 = arith.constant 1 : index
// CHECK-NEXT:        %218 = arith.addi %186, %217 : index
// CHECK-NEXT:        memref.store %194, %174[%214, %216, %218] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_index(%91 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %92 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x = stencil.index 0 [0, 0, 0]
      %y = stencil.index 1 [0, 0, 0]
      %z = stencil.index 2 [0, 0, 0]
      %xy = arith.addi %x, %y : index
      %xyz = arith.addi %xy, %z : index
      stencil.return %xyz : index
    }
    stencil.store %92 to %91 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// CHECK-NEXT:    func.func @stencil_init_index(%219 : memref<64x64x64xindex>) {
// CHECK-NEXT:      %220 = "memref.subview"(%219) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x64x64xindex>) -> memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:      %221 = arith.constant 0 : index
// CHECK-NEXT:      %222 = arith.constant 0 : index
// CHECK-NEXT:      %223 = arith.constant 0 : index
// CHECK-NEXT:      %224 = arith.constant 1 : index
// CHECK-NEXT:      %225 = arith.constant 1 : index
// CHECK-NEXT:      %226 = arith.constant 1 : index
// CHECK-NEXT:      %227 = arith.constant 64 : index
// CHECK-NEXT:      %228 = arith.constant 64 : index
// CHECK-NEXT:      %229 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%221, %222, %223, %227, %228, %229, %224, %225, %226) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^13(%x : index, %y : index, %z : index):
// CHECK-NEXT:        %xy = arith.addi %x, %y : index
// CHECK-NEXT:        %xyz = arith.addi %xy, %z : index
// CHECK-NEXT:        memref.store %xyz, %220[%x, %y, %z] : memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_index_offset(%93 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %94 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x_1 = stencil.index 0 [1, 1, 1]
      %y_1 = stencil.index 1 [-1, -1, -1]
      %z_1 = stencil.index 2 [0, 0, 0]
      %xy_1 = arith.addi %x_1, %y_1 : index
      %xyz_1 = arith.addi %xy_1, %z_1 : index
      stencil.return %xyz_1 : index
    }
    stencil.store %94 to %93 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// CHECK-NEXT:    func.func @stencil_init_index_offset(%230 : memref<64x64x64xindex>) {
// CHECK-NEXT:      %231 = "memref.subview"(%230) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x64x64xindex>) -> memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:      %232 = arith.constant 0 : index
// CHECK-NEXT:      %233 = arith.constant 0 : index
// CHECK-NEXT:      %234 = arith.constant 0 : index
// CHECK-NEXT:      %235 = arith.constant 1 : index
// CHECK-NEXT:      %236 = arith.constant 1 : index
// CHECK-NEXT:      %237 = arith.constant 1 : index
// CHECK-NEXT:      %238 = arith.constant 64 : index
// CHECK-NEXT:      %239 = arith.constant 64 : index
// CHECK-NEXT:      %240 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%232, %233, %234, %238, %239, %240, %235, %236, %237) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^14(%241 : index, %242 : index, %z_1 : index):
// CHECK-NEXT:        %x_1 = arith.constant 1 : index
// CHECK-NEXT:        %x_1_1 = arith.addi %241, %x_1 : index
// CHECK-NEXT:        %y_1 = arith.constant -1 : index
// CHECK-NEXT:        %y_1_1 = arith.addi %242, %y_1 : index
// CHECK-NEXT:        %xy_1 = arith.addi %x_1_1, %y_1_1 : index
// CHECK-NEXT:        %xyz_1 = arith.addi %xy_1, %z_1 : index
// CHECK-NEXT:        memref.store %xyz_1, %231[%241, %242, %z_1] : memref<64x64x64xindex, strided<[4096, 64, 1]>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @store_result_lowering(%arg0 : f64) {
    %95, %96 = stencil.apply(%arg1 = %arg0 : f64) -> (!stencil.temp<[0,7]x[0,7]x[0,7]xf64>, !stencil.temp<[0,7]x[0,7]x[0,7]xf64>) {
      %97 = stencil.store_result %arg1 : !stencil.result<f64>
      %98 = stencil.store_result %arg1 : !stencil.result<f64>
      stencil.return %97, %98 : !stencil.result<f64>, !stencil.result<f64>
    }
    %99 = stencil.buffer %96 : !stencil.temp<[0,7]x[0,7]x[0,7]xf64>
    %100 = stencil.buffer %95 : !stencil.temp<[0,7]x[0,7]x[0,7]xf64>
    func.return
  }

// CHECK-NEXT:    func.func @store_result_lowering(%arg0 : f64) {
// CHECK-NEXT:      %243 = memref.alloc() : memref<7x7x7xf64>
// CHECK-NEXT:      %244 = "memref.subview"(%243) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 7, 7, 7>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<7x7x7xf64>) -> memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %245 = memref.alloc() : memref<7x7x7xf64>
// CHECK-NEXT:      %246 = "memref.subview"(%245) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 7, 7, 7>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<7x7x7xf64>) -> memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %247 = arith.constant 0 : index
// CHECK-NEXT:      %248 = arith.constant 0 : index
// CHECK-NEXT:      %249 = arith.constant 0 : index
// CHECK-NEXT:      %250 = arith.constant 1 : index
// CHECK-NEXT:      %251 = arith.constant 1 : index
// CHECK-NEXT:      %252 = arith.constant 1 : index
// CHECK-NEXT:      %253 = arith.constant 7 : index
// CHECK-NEXT:      %254 = arith.constant 7 : index
// CHECK-NEXT:      %255 = arith.constant 7 : index
// CHECK-NEXT:      "scf.parallel"(%247, %248, %249, %253, %254, %255, %250, %251, %252) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^15(%256 : index, %257 : index, %258 : index):
// CHECK-NEXT:        memref.store %arg0, %246[%256, %257, %258] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:        memref.store %arg0, %244[%256, %257, %258] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      memref.dealloc %243 : memref<7x7x7xf64>
// CHECK-NEXT:      memref.dealloc %245 : memref<7x7x7xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @if_lowering(%arg0_1 : f64, %b0 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>, %b1 : !stencil.field<[0,7]x[0,7]x[0,7]xf64>)  attributes {"stencil.program"}{
    %101, %102 = stencil.apply(%arg1_1 = %arg0_1 : f64) -> (!stencil.temp<[0,7]x[0,7]x[0,7]xf64>, !stencil.temp<[0,7]x[0,7]x[0,7]xf64>) {
      %true = "test.op"() : () -> i1
      %103, %104 = "scf.if"(%true) ({
        %105 = stencil.store_result %arg1_1 : !stencil.result<f64>
        scf.yield %105, %arg1_1 : !stencil.result<f64>, f64
      }, {
        %106 = stencil.store_result  : !stencil.result<f64>
        scf.yield %106, %arg1_1 : !stencil.result<f64>, f64
      }) : (i1) -> (!stencil.result<f64>, f64)
      %107 = stencil.store_result %104 : !stencil.result<f64>
      stencil.return %103, %107 : !stencil.result<f64>, !stencil.result<f64>
    }
    stencil.store %101 to %b0 ([0, 0, 0] : [7, 7, 7]) : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> to !stencil.field<[0,7]x[0,7]x[0,7]xf64>
    stencil.store %102 to %b1 ([0, 0, 0] : [7, 7, 7]) : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> to !stencil.field<[0,7]x[0,7]x[0,7]xf64>
    func.return
  }

// CHECK-NEXT:    func.func @if_lowering(%arg0_1 : f64, %b0 : memref<7x7x7xf64>, %b1 : memref<7x7x7xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %b1_storeview = "memref.subview"(%b1) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 7, 7, 7>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<7x7x7xf64>) -> memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %b0_storeview = "memref.subview"(%b0) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 7, 7, 7>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<7x7x7xf64>) -> memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:      %259 = arith.constant 0 : index
// CHECK-NEXT:      %260 = arith.constant 0 : index
// CHECK-NEXT:      %261 = arith.constant 0 : index
// CHECK-NEXT:      %262 = arith.constant 1 : index
// CHECK-NEXT:      %263 = arith.constant 1 : index
// CHECK-NEXT:      %264 = arith.constant 1 : index
// CHECK-NEXT:      %265 = arith.constant 7 : index
// CHECK-NEXT:      %266 = arith.constant 7 : index
// CHECK-NEXT:      %267 = arith.constant 7 : index
// CHECK-NEXT:      "scf.parallel"(%259, %260, %261, %265, %266, %267, %262, %263, %264) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:      ^16(%268 : index, %269 : index, %270 : index):
// CHECK-NEXT:        %true = "test.op"() : () -> i1
// CHECK-NEXT:        %271, %272 = "scf.if"(%true) ({
// CHECK-NEXT:          memref.store %arg0_1, %b0_storeview[%268, %269, %270] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:          scf.yield %arg0_1, %arg0_1 : f64, f64
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %273 = builtin.unrealized_conversion_cast to f64
// CHECK-NEXT:          scf.yield %273, %arg0_1 : f64, f64
// CHECK-NEXT:        }) : (i1) -> (f64, f64)
// CHECK-NEXT:        memref.store %272, %b1_storeview[%268, %269, %270] : memref<7x7x7xf64, strided<[49, 7, 1]>>
// CHECK-NEXT:        scf.yield
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
    stencil.store %114 to %109 ([1, 2] : [65, 66]) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK-NEXT:    func.func @combine(%274 : memref<?x?xf64>) {
// CHECK-NEXT:      %275 = "memref.cast"(%274) : (memref<?x?xf64>) -> memref<70x70xf64>
// CHECK-NEXT:      %276 = "memref.subview"(%275) <{"static_offsets" = array<i64: 3, 3>, "static_sizes" = array<i64: 64, 64>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<70x70xf64>) -> memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:      %277 = arith.constant 1 : index
// CHECK-NEXT:      %278 = arith.constant 2 : index
// CHECK-NEXT:      %279 = arith.constant 1 : index
// CHECK-NEXT:      %280 = arith.constant 1 : index
// CHECK-NEXT:      %281 = arith.constant 33 : index
// CHECK-NEXT:      %282 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%277, %278, %281, %282, %279, %280) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^17(%283 : index, %284 : index):
// CHECK-NEXT:        %285 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        memref.store %285, %276[%283, %284] : memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %286 = arith.constant 33 : index
// CHECK-NEXT:      %287 = arith.constant 2 : index
// CHECK-NEXT:      %288 = arith.constant 1 : index
// CHECK-NEXT:      %289 = arith.constant 1 : index
// CHECK-NEXT:      %290 = arith.constant 65 : index
// CHECK-NEXT:      %291 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%286, %287, %290, %291, %288, %289) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^18(%292 : index, %293 : index):
// CHECK-NEXT:        %294 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        memref.store %294, %276[%292, %293] : memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:        scf.yield
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
    %122 = stencil.buffer %121 : !stencil.temp<[1,65]x[2,66]xf64>
    %123 = stencil.apply(%124 = %122 : !stencil.temp<[1,65]x[2,66]xf64>) -> (!stencil.temp<[1,65]x[2,66]xf64>) {
      %125 = arith.constant 1.000000e+00 : f64
      %126 = stencil.access %124 [0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
      %127 = arith.addf %125, %126 : f64
      stencil.return %127 : f64
    }
    stencil.store %123 to %116 ([1, 2] : [65, 66]) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK-NEXT:    func.func @buffered_combine(%295 : memref<?x?xf64>) {
// CHECK-NEXT:      %296 = memref.alloc() : memref<64x64xf64>
// CHECK-NEXT:      %297 = "memref.subview"(%296) <{"static_offsets" = array<i64: -1, -2>, "static_sizes" = array<i64: 64, 64>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x64xf64>) -> memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:      %298 = "memref.cast"(%295) : (memref<?x?xf64>) -> memref<70x70xf64>
// CHECK-NEXT:      %299 = "memref.subview"(%298) <{"static_offsets" = array<i64: 3, 3>, "static_sizes" = array<i64: 64, 64>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<70x70xf64>) -> memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:      %300 = arith.constant 1 : index
// CHECK-NEXT:      %301 = arith.constant 2 : index
// CHECK-NEXT:      %302 = arith.constant 1 : index
// CHECK-NEXT:      %303 = arith.constant 1 : index
// CHECK-NEXT:      %304 = arith.constant 33 : index
// CHECK-NEXT:      %305 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%300, %301, %304, %305, %302, %303) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^19(%306 : index, %307 : index):
// CHECK-NEXT:        %308 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        memref.store %308, %297[%306, %307] : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %309 = arith.constant 33 : index
// CHECK-NEXT:      %310 = arith.constant 2 : index
// CHECK-NEXT:      %311 = arith.constant 1 : index
// CHECK-NEXT:      %312 = arith.constant 1 : index
// CHECK-NEXT:      %313 = arith.constant 65 : index
// CHECK-NEXT:      %314 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%309, %310, %313, %314, %311, %312) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^20(%315 : index, %316 : index):
// CHECK-NEXT:        %317 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        memref.store %317, %297[%315, %316] : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %318 = arith.constant 1 : index
// CHECK-NEXT:      %319 = arith.constant 2 : index
// CHECK-NEXT:      %320 = arith.constant 1 : index
// CHECK-NEXT:      %321 = arith.constant 1 : index
// CHECK-NEXT:      %322 = arith.constant 65 : index
// CHECK-NEXT:      %323 = arith.constant 66 : index
// CHECK-NEXT:      "scf.parallel"(%318, %319, %322, %323, %320, %321) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^21(%324 : index, %325 : index):
// CHECK-NEXT:        %326 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %327 = memref.load %297[%324, %325] : memref<64x64xf64, strided<[64, 1], offset: -66>>
// CHECK-NEXT:        %328 = arith.addf %326, %327 : f64
// CHECK-NEXT:        memref.store %328, %299[%324, %325] : memref<64x64xf64, strided<[70, 1], offset: 213>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      memref.dealloc %296 : memref<64x64xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offset_mapping(%0 : !stencil.field<[0,8]xf64>, %1 : !stencil.field<[0,8]xf64>, %2 : !stencil.field<[0,8]x[0,8]xf64>) {
    %3 = "stencil.load"(%0) : (!stencil.field<[0,8]xf64>) -> !stencil.temp<[0,8]xf64>
    %4 = "stencil.load"(%1) : (!stencil.field<[0,8]xf64>) -> !stencil.temp<[0,8]xf64>
    %5 = "stencil.apply"(%3, %4) ({
    ^0(%6 :  !stencil.temp<[0,8]xf64>, %10 :  !stencil.temp<[0,8]xf64>):
    %7 = "stencil.access"(%6) {"offset" = #stencil.index<0>, "offset_mapping" = #stencil.index<0>} : (!stencil.temp<[0,8]xf64>) -> f64
    %8 = "stencil.access"(%10) {"offset" = #stencil.index<0>, "offset_mapping" = #stencil.index<1>} : (!stencil.temp<[0,8]xf64>) -> f64
    %9 = arith.mulf %7, %8 : f64
    "stencil.return"(%9) : (f64) -> ()
    }) : (!stencil.temp<[0,8]xf64>, !stencil.temp<[0,8]xf64>) -> !stencil.temp<[0,8]x[0,8]xf64>
    "stencil.store"(%5, %2) {"lb" = #stencil.index<0>, "ub" = #stencil.index<8>} : (!stencil.temp<[0,8]x[0,8]xf64>, !stencil.field<[0,8]x[0,8]xf64>) -> ()
    func.return
  }

// CHECK-NEXT:    func.func @offset_mapping(%329 : memref<8xf64>, %330 : memref<8xf64>, %331 : memref<8x8xf64>) {
// CHECK-NEXT:      %332 = "memref.subview"(%331) <{"static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 8, 8>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<8x8xf64>) -> memref<8x8xf64, strided<[8, 1]>>
// CHECK-NEXT:      %333 = "memref.subview"(%329) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 8>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<8xf64>) -> memref<8xf64, strided<[1]>>
// CHECK-NEXT:      %334 = "memref.subview"(%330) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 8>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<8xf64>) -> memref<8xf64, strided<[1]>>
// CHECK-NEXT:      %335 = arith.constant 0 : index
// CHECK-NEXT:      %336 = arith.constant 0 : index
// CHECK-NEXT:      %337 = arith.constant 1 : index
// CHECK-NEXT:      %338 = arith.constant 1 : index
// CHECK-NEXT:      %339 = arith.constant 8 : index
// CHECK-NEXT:      %340 = arith.constant 8 : index
// CHECK-NEXT:      "scf.parallel"(%335, %336, %339, %340, %337, %338) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:      ^22(%341 : index, %342 : index):
// CHECK-NEXT:        %343 = memref.load %333[%341] : memref<8xf64, strided<[1]>>
// CHECK-NEXT:        %344 = memref.load %334[%342] : memref<8xf64, strided<[1]>>
// CHECK-NEXT:        %345 = arith.mulf %343, %344 : f64
// CHECK-NEXT:        memref.store %345, %332[%341, %342] : memref<8x8xf64, strided<[8, 1]>>
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
  

}
// CHECK-NEXT: }
