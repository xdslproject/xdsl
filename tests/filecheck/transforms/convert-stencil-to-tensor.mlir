// RUN: xdsl-opt %s -p convert-stencil-to-tensor | filecheck %s

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

  // CHECK-NEXT:    func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
  // CHECK-NEXT:      %3 = "bufferization.to_tensor"(%2) <{"restrict", "writable"}> : (memref<70x70x70xf64>) -> tensor<70x70x70xf64>
  // CHECK-NEXT:      %4 = tensor.empty() : tensor<64x64x60xf64>
  // CHECK-NEXT:      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"], doc = "apply"} outs(%4 : tensor<64x64x60xf64>) {
  // CHECK-NEXT:      ^0(%6 : f64):
  // CHECK-NEXT:        %7 = arith.constant 1.000000e+00 : f64
  // CHECK-NEXT:        %8 = arith.addf %0, %7 : f64
  // CHECK-NEXT:        linalg.yield %8 : f64
  // CHECK-NEXT:      } -> tensor<64x64x60xf64>
  // CHECK-NEXT:      %9 = "tensor.insert_slice"(%5, %3) <{"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64x64x60xf64>, tensor<70x70x70xf64>) -> tensor<70x70x70xf64>
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

  // CHECK-NEXT:    func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
  // CHECK-NEXT:      %time_m = arith.constant 0 : index
  // CHECK-NEXT:      %time_M = arith.constant 1001 : index
  // CHECK-NEXT:      %step = arith.constant 1 : index
  // CHECK-NEXT:      %t1_out, %t0_out = scf.for %time = %time_m to %time_M step %step iter_args(%fim1 = %f0, %fi = %f1) -> (memref<2004x2004xf32>, memref<2004x2004xf32>) {
  // CHECK-NEXT:        %10 = "bufferization.to_tensor"(%fi) <{"restrict", "writable"}> : (memref<2004x2004xf32>) -> tensor<2004x2004xf32>
  // CHECK-NEXT:        %tim1 = "bufferization.to_tensor"(%fim1) <{"restrict", "writable"}> : (memref<2004x2004xf32>) -> tensor<2004x2004xf32>
  // CHECK-NEXT:        %tim1_1 = "tensor.extract_slice"(%tim1) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<2004x2004xf32>) -> tensor<2000x2000xf32>
  // CHECK-NEXT:        %ti = "tensor.extract_slice"(%tim1_1) <{"static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<2000x2000xf32>) -> tensor<2000x2000xf32>
  // CHECK-NEXT:        %ti_1 = tensor.empty() : tensor<2000x2000xf32>
  // CHECK-NEXT:        %ti_2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc = "apply"} ins(%ti : tensor<2000x2000xf32>) outs(%ti_1 : tensor<2000x2000xf32>) {
  // CHECK-NEXT:        ^1(%i : f32, %11 : f32):
  // CHECK-NEXT:          linalg.yield %i : f32
  // CHECK-NEXT:        } -> tensor<2000x2000xf32>
  // CHECK-NEXT:        %12 = "tensor.insert_slice"(%ti_2, %10) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<2000x2000xf32>, tensor<2004x2004xf32>) -> tensor<2004x2004xf32>
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

  // CHECK-NEXT:    func.func @copy_1d(%13 : memref<?xf64>, %out : memref<?xf64>) {
  // CHECK-NEXT:      %14 = "memref.cast"(%13) : (memref<?xf64>) -> memref<72xf64>
  // CHECK-NEXT:      %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
  // CHECK-NEXT:      %15 = "bufferization.to_tensor"(%outc) <{"restrict", "writable"}> : (memref<1024xf64>) -> tensor<1024xf64>
  // CHECK-NEXT:      %16 = "bufferization.to_tensor"(%14) <{"restrict", "writable"}> : (memref<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %17 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 3>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<72xf64>) -> tensor<69xf64>
  // CHECK-NEXT:      %18 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<69xf64>) -> tensor<68xf64>
  // CHECK-NEXT:      %19 = tensor.empty() : tensor<68xf64>
  // CHECK-NEXT:      %20 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], doc = "apply"} ins(%18 : tensor<68xf64>) outs(%19 : tensor<68xf64>) {
  // CHECK-NEXT:      ^2(%21 : f64, %22 : f64):
  // CHECK-NEXT:        linalg.yield %21 : f64
  // CHECK-NEXT:      } -> tensor<68xf64>
  // CHECK-NEXT:      %23 = "tensor.insert_slice"(%20, %15) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<68xf64>, tensor<1024xf64>) -> tensor<1024xf64>
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

  // CHECK-NEXT:    func.func @copy_2d(%24 : memref<?x?xf64>) {
  // CHECK-NEXT:      %25 = "memref.cast"(%24) : (memref<?x?xf64>) -> memref<72x72xf64>
  // CHECK-NEXT:      %26 = "bufferization.to_tensor"(%25) <{"restrict", "writable"}> : (memref<72x72xf64>) -> tensor<72x72xf64>
  // CHECK-NEXT:      %27 = "tensor.extract_slice"(%26) <{"static_offsets" = array<i64: 3, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<72x72xf64>) -> tensor<65x68xf64>
  // CHECK-NEXT:      %28 = "tensor.extract_slice"(%27) <{"static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 64, 68>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<65x68xf64>) -> tensor<64x68xf64>
  // CHECK-NEXT:      %29 = tensor.empty() : tensor<64x68xf64>
  // CHECK-NEXT:      %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc = "apply"} ins(%28 : tensor<64x68xf64>) outs(%29 : tensor<64x68xf64>) {
  // CHECK-NEXT:      ^3(%31 : f64, %32 : f64):
  // CHECK-NEXT:        linalg.yield %31 : f64
  // CHECK-NEXT:      } -> tensor<64x68xf64>
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

  // CHECK-NEXT:    func.func @copy_3d(%33 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      %34 = "memref.cast"(%33) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
  // CHECK-NEXT:      %35 = "bufferization.to_tensor"(%34) <{"restrict", "writable"}> : (memref<72x74x76xf64>) -> tensor<72x74x76xf64>
  // CHECK-NEXT:      %36 = "tensor.extract_slice"(%35) <{"static_offsets" = array<i64: 3, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<72x74x76xf64>) -> tensor<65x64x69xf64>
  // CHECK-NEXT:      %37 = "tensor.extract_slice"(%36) <{"static_offsets" = array<i64: 0, 0, 1>, "static_sizes" = array<i64: 64, 64, 68>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<65x64x69xf64>) -> tensor<64x64x68xf64>
  // CHECK-NEXT:      %38 = tensor.empty() : tensor<64x64x68xf64>
  // CHECK-NEXT:      %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"], doc = "apply"} ins(%37 : tensor<64x64x68xf64>) outs(%38 : tensor<64x64x68xf64>) {
  // CHECK-NEXT:      ^4(%40 : f64, %41 : f64):
  // CHECK-NEXT:        linalg.yield %40 : f64
  // CHECK-NEXT:      } -> tensor<64x64x68xf64>
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }

  // CHECK-NEXT:    func.func @test_funcop_lowering(%42 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }
  // CHECK-NEXT:    func.func @test_funcop_lowering_dyn(%43 : memref<8x8xf64>) {
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

  // CHECK-NEXT:    func.func @offsets(%44 : memref<?x?x?xf64>, %45 : memref<?x?x?xf64>, %46 : memref<?x?x?xf64>) {
  // CHECK-NEXT:      %47 = "memref.cast"(%44) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:      %48 = "memref.cast"(%45) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:      %49 = "bufferization.to_tensor"(%48) <{"restrict", "writable"}> : (memref<72x72x72xf64>) -> tensor<72x72x72xf64>
  // CHECK-NEXT:      %50 = "memref.cast"(%46) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
  // CHECK-NEXT:      %51 = "bufferization.to_tensor"(%50) <{"restrict", "writable"}> : (memref<72x72x72xf64>) -> tensor<72x72x72xf64>
  // CHECK-NEXT:      %52 = "bufferization.to_tensor"(%47) <{"restrict", "writable"}> : (memref<72x72x72xf64>) -> tensor<72x72x72xf64>
  // CHECK-NEXT:      %53 = "tensor.extract_slice"(%52) <{"static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<72x72x72xf64>) -> tensor<66x66x64xf64>
  // CHECK-NEXT:      %54 = "tensor.extract_slice"(%53) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<66x66x64xf64>) -> tensor<64x64x64xf64>
  // CHECK-NEXT:      %55 = "tensor.extract_slice"(%53) <{"static_offsets" = array<i64: 2, 1, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<66x66x64xf64>) -> tensor<64x64x64xf64>
  // CHECK-NEXT:      %56 = "tensor.extract_slice"(%53) <{"static_offsets" = array<i64: 1, 2, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<66x66x64xf64>) -> tensor<64x64x64xf64>
  // CHECK-NEXT:      %57 = "tensor.extract_slice"(%53) <{"static_offsets" = array<i64: 1, 0, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<66x66x64xf64>) -> tensor<64x64x64xf64>
  // CHECK-NEXT:      %58 = "tensor.extract_slice"(%53) <{"static_offsets" = array<i64: 1, 1, 0>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<66x66x64xf64>) -> tensor<64x64x64xf64>
  // CHECK-NEXT:      %59 = tensor.empty() : tensor<64x64x64xf64>
  // CHECK-NEXT:      %60 = tensor.empty() : tensor<64x64x64xf64>
  // CHECK-NEXT:      %61, %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"], doc = "apply"} ins(%54, %55, %56, %57, %58 : tensor<64x64x64xf64>, tensor<64x64x64xf64>, tensor<64x64x64xf64>, tensor<64x64x64xf64>, tensor<64x64x64xf64>) outs(%59, %60 : tensor<64x64x64xf64>, tensor<64x64x64xf64>) {
  // CHECK-NEXT:      ^5(%63 : f64, %64 : f64, %65 : f64, %66 : f64, %67 : f64, %68 : f64, %69 : f64):
  // CHECK-NEXT:        %70 = arith.addf %63, %64 : f64
  // CHECK-NEXT:        %71 = arith.addf %65, %66 : f64
  // CHECK-NEXT:        %72 = arith.addf %70, %71 : f64
  // CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
  // CHECK-NEXT:        %73 = arith.mulf %67, %cst : f64
  // CHECK-NEXT:        %74 = arith.addf %73, %72 : f64
  // CHECK-NEXT:        linalg.yield %74, %73 : f64, f64
  // CHECK-NEXT:      } -> (tensor<64x64x64xf64>, tensor<64x64x64xf64>)
  // CHECK-NEXT:      %75 = "tensor.insert_slice"(%61, %49) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64x64x64xf64>, tensor<72x72x72xf64>) -> tensor<72x72x72xf64>
  // CHECK-NEXT:      %76 = "tensor.insert_slice"(%62, %51) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64x64x64xf64>, tensor<72x72x72xf64>) -> tensor<72x72x72xf64>
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

  // CHECK-NEXT:    func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : memref<?x?x?xf64>, %sta_field : memref<64x64x64xf64>) {
  // CHECK-NEXT:      %77 = builtin.unrealized_conversion_cast %dyn_mem : memref<?x?x?xf64> to memref<?x?x?xf64>
  // CHECK-NEXT:      %78 = builtin.unrealized_conversion_cast %sta_mem : memref<64x64x64xf64> to memref<64x64x64xf64>
  // CHECK-NEXT:      %casted = "memref.cast"(%77) : (memref<?x?x?xf64>) -> memref<64x64x64xf64>
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

  // CHECK-NEXT:    func.func @neg_bounds(%in : memref<64xf64>, %out_1 : memref<64xf64>) {
  // CHECK-NEXT:      %79 = "bufferization.to_tensor"(%out_1) <{"restrict", "writable"}> : (memref<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %tin = "bufferization.to_tensor"(%in) <{"restrict", "writable"}> : (memref<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %tin_1 = "tensor.extract_slice"(%tin) <{"static_offsets" = array<i64: 16>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf64>) -> tensor<32xf64>
  // CHECK-NEXT:      %outt = "tensor.extract_slice"(%tin_1) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<32xf64>) -> tensor<32xf64>
  // CHECK-NEXT:      %outt_1 = tensor.empty() : tensor<32xf64>
  // CHECK-NEXT:      %outt_2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], doc = "apply"} ins(%outt : tensor<32xf64>) outs(%outt_1 : tensor<32xf64>) {
  // CHECK-NEXT:      ^6(%val : f64, %80 : f64):
  // CHECK-NEXT:        linalg.yield %val : f64
  // CHECK-NEXT:      } -> tensor<32xf64>
  // CHECK-NEXT:      %81 = "tensor.insert_slice"(%outt_2, %79) <{"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<32xf64>, tensor<64xf64>) -> tensor<64xf64>
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

  // CHECK-NEXT:    func.func @stencil_buffer(%82 : memref<72xf64>, %83 : memref<72xf64>) {
  // CHECK-NEXT:      %84 = "bufferization.to_tensor"(%83) <{"restrict", "writable"}> : (memref<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %85 = "bufferization.to_tensor"(%82) <{"restrict", "writable"}> : (memref<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %86 = "tensor.extract_slice"(%85) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<72xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %87 = "bufferization.alloc_tensor"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> tensor<64xf64>
  // CHECK-NEXT:      %88 = "tensor.extract_slice"(%86) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %89 = tensor.empty() : tensor<64xf64>
  // CHECK-NEXT:      %90 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], doc = "apply"} ins(%88 : tensor<64xf64>) outs(%89 : tensor<64xf64>) {
  // CHECK-NEXT:      ^7(%91 : f64, %92 : f64):
  // CHECK-NEXT:        linalg.yield %91 : f64
  // CHECK-NEXT:      } -> tensor<64xf64>
  // CHECK-NEXT:      %93 = "tensor.insert_slice"(%90, %87) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64xf64>, tensor<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %94 = "tensor.extract_slice"(%93) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %95 = tensor.empty() : tensor<64xf64>
  // CHECK-NEXT:      %96 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], doc = "apply"} ins(%94 : tensor<64xf64>) outs(%95 : tensor<64xf64>) {
  // CHECK-NEXT:      ^8(%97 : f64, %98 : f64):
  // CHECK-NEXT:        linalg.yield %97 : f64
  // CHECK-NEXT:      } -> tensor<64xf64>
  // CHECK-NEXT:      %99 = "tensor.insert_slice"(%96, %84) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64xf64>, tensor<72xf64>) -> tensor<72xf64>
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

  // CHECK-NEXT:    func.func @stencil_two_stores(%100 : memref<72xf64>, %101 : memref<72xf64>, %102 : memref<72xf64>) {
  // CHECK-NEXT:      %103 = "bufferization.to_tensor"(%102) <{"restrict", "writable"}> : (memref<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %104 = "bufferization.to_tensor"(%101) <{"restrict", "writable"}> : (memref<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %105 = "bufferization.to_tensor"(%100) <{"restrict", "writable"}> : (memref<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %106 = "tensor.extract_slice"(%105) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<72xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %107 = "tensor.extract_slice"(%106) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %108 = tensor.empty() : tensor<64xf64>
  // CHECK-NEXT:      %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], doc = "apply"} ins(%107 : tensor<64xf64>) outs(%108 : tensor<64xf64>) {
  // CHECK-NEXT:      ^9(%110 : f64, %111 : f64):
  // CHECK-NEXT:        linalg.yield %110 : f64
  // CHECK-NEXT:      } -> tensor<64xf64>
  // CHECK-NEXT:      %112 = "tensor.insert_slice"(%109, %103) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64xf64>, tensor<72xf64>) -> tensor<72xf64>
  // CHECK-NEXT:      %113 = "tensor.extract_slice"(%109) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf64>) -> tensor<64xf64>
  // CHECK-NEXT:      %114 = tensor.empty() : tensor<64xf64>
  // CHECK-NEXT:      %115 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], doc = "apply"} ins(%113 : tensor<64xf64>) outs(%114 : tensor<64xf64>) {
  // CHECK-NEXT:      ^10(%116 : f64, %117 : f64):
  // CHECK-NEXT:        linalg.yield %116 : f64
  // CHECK-NEXT:      } -> tensor<64xf64>
  // CHECK-NEXT:      %118 = "tensor.insert_slice"(%115, %104) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<64xf64>, tensor<72xf64>) -> tensor<72xf64>
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
  
  // CHECK-NEXT:    func.func @apply_kernel(%119 : memref<15x15xf32>, %120 : memref<15x15xf32>, %timers : !llvm.ptr)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
  // CHECK-NEXT:      %121 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
  // CHECK-NEXT:      %u_vec_1 = builtin.unrealized_conversion_cast %121 : memref<15x15xf32> to memref<15x15xf32>
  // CHECK-NEXT:      %122 = builtin.unrealized_conversion_cast %120 : memref<15x15xf32> to memref<15x15xf32>
  // CHECK-NEXT:      "gpu.memcpy"(%121, %122) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
  // CHECK-NEXT:      %123 = "gpu.alloc"() <{"operandSegmentSizes" = array<i32: 0, 0, 0>}> : () -> memref<15x15xf32>
  // CHECK-NEXT:      %u_vec_0 = builtin.unrealized_conversion_cast %123 : memref<15x15xf32> to memref<15x15xf32>
  // CHECK-NEXT:      %124 = builtin.unrealized_conversion_cast %119 : memref<15x15xf32> to memref<15x15xf32>
  // CHECK-NEXT:      "gpu.memcpy"(%123, %124) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
  // CHECK-NEXT:      %time_m_1 = arith.constant 0 : index
  // CHECK-NEXT:      %time_M_1 = arith.constant 10 : index
  // CHECK-NEXT:      %step_1 = arith.constant 1 : index
  // CHECK-NEXT:      %125, %126 = scf.for %time_1 = %time_m_1 to %time_M_1 step %step_1 iter_args(%t0 = %u_vec_0, %t1 = %u_vec_1) -> (memref<15x15xf32>, memref<15x15xf32>) {
  // CHECK-NEXT:        %127 = "bufferization.to_tensor"(%t1) <{"restrict", "writable"}> : (memref<15x15xf32>) -> tensor<15x15xf32>
  // CHECK-NEXT:        %t0_temp = "bufferization.to_tensor"(%t0) <{"restrict", "writable"}> : (memref<15x15xf32>) -> tensor<15x15xf32>
  // CHECK-NEXT:        %t0_temp_1 = "tensor.extract_slice"(%t0_temp) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<15x15xf32>) -> tensor<11x11xf32>
  // CHECK-NEXT:        %t1_result = "tensor.extract_slice"(%t0_temp_1) <{"static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<11x11xf32>) -> tensor<11x11xf32>
  // CHECK-NEXT:        %t1_result_1 = tensor.empty() : tensor<11x11xf32>
  // CHECK-NEXT:        %t1_result_2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc = "apply"} ins(%t1_result : tensor<11x11xf32>) outs(%t1_result_1 : tensor<11x11xf32>) {
  // CHECK-NEXT:        ^11(%128 : f32, %129 : f32):
  // CHECK-NEXT:          linalg.yield %128 : f32
  // CHECK-NEXT:        } -> tensor<11x11xf32>
  // CHECK-NEXT:        %130 = "tensor.insert_slice"(%t1_result_2, %127) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> : (tensor<11x11xf32>, tensor<15x15xf32>) -> tensor<15x15xf32>
  // CHECK-NEXT:        scf.yield %t1, %t0 : memref<15x15xf32>, memref<15x15xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:      func.return
  // CHECK-NEXT:    }

}
// CHECK-NEXT:  }
