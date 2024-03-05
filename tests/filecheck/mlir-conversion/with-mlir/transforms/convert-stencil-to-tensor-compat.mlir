// RUN: xdsl-opt %s -p convert-stencil-to-tensor-compat | filecheck %s

builtin.module {
// CHECK:       builtin.module {

  func.func private @external(!stencil.field<?xf64>) -> ()
  // CHECK-NEXT:    func.func private @external(memref<?xf64>) -> ()

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

  // CHECK-NEXT:    func.func @stencil_init_float(%arg0 : f64, %arg1 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      %0 = arith.constant 0 : index
  // CHECK-NEXT:      %1 = arith.constant 64 : index
  // CHECK-NEXT:      %2 = arith.constant 1 : index
  // CHECK-NEXT:      %3 = arith.constant 60 : index
  // CHECK-NEXT:      %4 = arith.constant 1.000000e+00 : f64
  // CHECK-NEXT:      %5 = "memref.cast"(%arg1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
  // CHECK-NEXT:      %6 = "memref.subview"(%5) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
  // CHECK-NEXT:      "scf.parallel"(%0, %0, %0, %1, %1, %3, %2, %2, %2) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
  // CHECK-NEXT:      ^0(%arg2 : index, %arg3 : index, %arg4 : index):
  // CHECK-NEXT:        %7 = arith.addf %arg0, %4 : f64
  // CHECK-NEXT:        memref.store %7, %6[%arg2, %arg3, %arg4] {"nontemporal" = false} : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
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

  // CHECK-NEXT:    func.func @bufferswapping(%arg0_1 : memref<2004x2004xf32>, %arg1_1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
  // CHECK-NEXT:      %8 = arith.constant 2000 : index
  // CHECK-NEXT:      %9 = arith.constant 0 : index
  // CHECK-NEXT:      %10 = arith.constant 1001 : index
  // CHECK-NEXT:      %11 = arith.constant 1 : index
  // CHECK-NEXT:      %12, %13 = scf.for %arg2_1 = %9 to %10 step %11 iter_args(%arg3_1 = %arg0_1, %arg4_1 = %arg1_1) -> (memref<2004x2004xf32>, memref<2004x2004xf32>) {
  // CHECK-NEXT:        %14 = "memref.subview"(%arg3_1) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>}> : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
  // CHECK-NEXT:        %15 = "memref.subview"(%arg4_1) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>}> : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
  // CHECK-NEXT:        "scf.parallel"(%9, %9, %8, %8, %11, %11) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
  // CHECK-NEXT:        ^1(%arg5 : index, %arg6 : index):
  // CHECK-NEXT:          %16 = memref.load %14[%arg5, %arg6] {"nontemporal" = false} : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
  // CHECK-NEXT:          memref.store %16, %15[%arg5, %arg6] {"nontemporal" = false} : memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
  // CHECK-NEXT:          scf.yield
  // CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
  // CHECK-NEXT:        scf.yield %arg4_1, %arg3_1 : memref<2004x2004xf32>, memref<2004x2004xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:      func.return %12 : memref<2004x2004xf32>
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

  // CHECK-NEXT:    func.func @copy_1d(%arg0_2 : memref<?xf64>, %arg1_2 : memref<?xf64>) {
  // CHECK-NEXT:      %17 = arith.constant 0 : index
  // CHECK-NEXT:      %18 = arith.constant 68 : index
  // CHECK-NEXT:      %19 = arith.constant 1 : index
  // CHECK-NEXT:      %20 = "memref.cast"(%arg0_2) : (memref<?xf64>) -> memref<72xf64>
  // CHECK-NEXT:      %21 = "memref.cast"(%arg1_2) : (memref<?xf64>) -> memref<1024xf64>
  // CHECK-NEXT:      %22 = "memref.subview"(%20) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 3>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>}> : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 3>>
  // CHECK-NEXT:      %23 = "memref.subview"(%22) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>}> : (memref<69xf64, strided<[1], offset: 3>>) -> memref<68xf64, strided<[1], offset: 3>>
  // CHECK-NEXT:      %24 = "memref.subview"(%21) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>}> : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
  // CHECK-NEXT:      "scf.parallel"(%17, %18, %19) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
  // CHECK-NEXT:      ^2(%arg2_2 : index):
  // CHECK-NEXT:        %25 = memref.load %23[%arg2_2] {"nontemporal" = false} : memref<68xf64, strided<[1], offset: 3>>
  // CHECK-NEXT:        memref.store %25, %24[%arg2_2] {"nontemporal" = false} : memref<68xf64, strided<[1]>>
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }) : (index, index, index) -> ()
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }

  // Kept this one as-is for demonstration;
  // This stencil IR actually never stores; so it's a no-op
  // This is optimized away by MLIR in this new flow!            
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

  // CHECK-NEXT:    func.func @copy_2d(%arg0_3 : memref<?x?xf64>) {
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }

                        
  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }

  // CHECK-NEXT:    func.func @test_funcop_lowering(%arg0_4 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }
  // CHECK-NEXT:    func.func @test_funcop_lowering_dyn(%arg0_5 : memref<8x8xf64>) {
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
    "stencil.store"(%8, %5) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }

  // CHECK-NEXT:    func.func @offsets(%arg0_6 : memref<?x?x?xf64>, %arg1_3 : memref<?x?x?xf64>, %arg2_3 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      %26 = arith.constant 0 : index
  // CHECK-NEXT:      %27 = arith.constant 64 : index
  // CHECK-NEXT:      %28 = arith.constant 1 : index
  // CHECK-NEXT:      %29 = arith.constant -4.000000e+00 : f64
  // CHECK-NEXT:      %30 = "memref.cast"(%arg0_6) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:      %31 = "memref.cast"(%arg1_3) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:      %32 = "memref.cast"(%arg2_3) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:      %33 = "memref.subview"(%30) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>
  // CHECK-NEXT:      %34 = "memref.subview"(%33) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 15844>>
  // CHECK-NEXT:      %35 = "memref.subview"(%33) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 2, 1, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 26212>>
  // CHECK-NEXT:      %36 = "memref.subview"(%33) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 1, 2, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21100>>
  // CHECK-NEXT:      %37 = "memref.subview"(%33) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 1, 0, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 20956>>
  // CHECK-NEXT:      %38 = "memref.subview"(%33) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 1, 1, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:      %39 = "memref.subview"(%31) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:      %40 = "memref.subview"(%32) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:      "scf.parallel"(%26, %26, %26, %27, %27, %27, %28, %28, %28) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
  // CHECK-NEXT:      ^3(%arg3_2 : index, %arg4_2 : index, %arg5_1 : index):
  // CHECK-NEXT:        %41 = memref.load %34[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 15844>>
  // CHECK-NEXT:        %42 = memref.load %35[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 26212>>
  // CHECK-NEXT:        %43 = memref.load %36[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21100>>
  // CHECK-NEXT:        %44 = memref.load %37[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 20956>>
  // CHECK-NEXT:        %45 = memref.load %38[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:        %46 = arith.addf %41, %42 : f64
  // CHECK-NEXT:        %47 = arith.addf %43, %44 : f64
  // CHECK-NEXT:        %48 = arith.addf %46, %47 : f64
  // CHECK-NEXT:        %49 = arith.mulf %45, %29 : f64
  // CHECK-NEXT:        %50 = arith.addf %49, %48 : f64
  // CHECK-NEXT:        memref.store %50, %39[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
  // CHECK-NEXT:        memref.store %49, %40[%arg3_2, %arg4_2, %arg5_1] {"nontemporal" = false} : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
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

  // CHECK-NEXT:    func.func @trivial_externals(%arg0_7 : memref<?x?x?xf64>, %arg1_4 : memref<64x64x64xf64>, %arg2_4 : memref<?x?x?xf64>, %arg3_3 : memref<64x64x64xf64>) {
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

  // CHECK-NEXT:    func.func @neg_bounds(%arg0_8 : memref<64xf64>, %arg1_5 : memref<64xf64>) {
  // CHECK-NEXT:      %51 = arith.constant 0 : index
  // CHECK-NEXT:      %52 = arith.constant 32 : index
  // CHECK-NEXT:      %53 = arith.constant 1 : index
  // CHECK-NEXT:      %54 = "memref.subview"(%arg0_8) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 16>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>}> : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 16>>
  // CHECK-NEXT:      %55 = "memref.subview"(%arg1_5) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>}> : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
  // CHECK-NEXT:      "scf.parallel"(%51, %52, %53) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
  // CHECK-NEXT:      ^4(%arg2_5 : index):
  // CHECK-NEXT:        %56 = memref.load %54[%arg2_5] {"nontemporal" = false} : memref<32xf64, strided<[1], offset: 16>>
  // CHECK-NEXT:        memref.store %56, %55[%arg2_5] {"nontemporal" = false} : memref<32xf64, strided<[1], offset: 32>>
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

  // CHECK-NEXT:    func.func @stencil_buffer(%arg0_9 : memref<72xf64>, %arg1_6 : memref<72xf64>) {
  // CHECK-NEXT:      %57 = arith.constant 0 : index
  // CHECK-NEXT:      %58 = arith.constant 64 : index
  // CHECK-NEXT:      %59 = arith.constant 1 : index
  // CHECK-NEXT:      %60 = "memref.subview"(%arg0_9) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:      %61 = memref.alloc() {"alignment" = 64 : i64} : memref<64xf64>
  // CHECK-NEXT:      "scf.parallel"(%57, %58, %59) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
  // CHECK-NEXT:      ^5(%arg2_6 : index):
  // CHECK-NEXT:        %62 = memref.load %60[%arg2_6] {"nontemporal" = false} : memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:        memref.store %62, %61[%arg2_6] {"nontemporal" = false} : memref<64xf64>
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }) : (index, index, index) -> ()
  // CHECK-NEXT:      %63 = "memref.subview"(%arg1_6) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:      "scf.parallel"(%57, %58, %59) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
  // CHECK-NEXT:      ^6(%arg2_7 : index):
  // CHECK-NEXT:        %64 = memref.load %61[%arg2_7] {"nontemporal" = false} : memref<64xf64>
  // CHECK-NEXT:        memref.store %64, %63[%arg2_7] {"nontemporal" = false} : memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }) : (index, index, index) -> ()
  // CHECK-NEXT:      memref.dealloc %61 : memref<64xf64>
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

  // CHECK-NEXT:    func.func @stencil_two_stores(%arg0_10 : memref<72xf64>, %arg1_7 : memref<72xf64>, %arg2_8 : memref<72xf64>) {
  // CHECK-NEXT:      %65 = arith.constant 0 : index
  // CHECK-NEXT:      %66 = arith.constant 64 : index
  // CHECK-NEXT:      %67 = arith.constant 1 : index
  // CHECK-NEXT:      %68 = "memref.subview"(%arg0_10) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:      %69 = "memref.subview"(%arg2_8) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:      "scf.parallel"(%65, %66, %67) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
  // CHECK-NEXT:      ^7(%arg3_4 : index):
  // CHECK-NEXT:        %70 = memref.load %68[%arg3_4] {"nontemporal" = false} : memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:        memref.store %70, %69[%arg3_4] {"nontemporal" = false} : memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }) : (index, index, index) -> ()
  // CHECK-NEXT:      %71 = "memref.subview"(%arg1_7) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:      "scf.parallel"(%65, %66, %67) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
  // CHECK-NEXT:      ^8(%arg3_5 : index):
  // CHECK-NEXT:        %72 = memref.load %69[%arg3_5] {"nontemporal" = false} : memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:        memref.store %72, %71[%arg3_5] {"nontemporal" = false} : memref<64xf64, strided<[1], offset: 4>>
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }) : (index, index, index) -> ()
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }          

}
// CHECK-NEXT: }
