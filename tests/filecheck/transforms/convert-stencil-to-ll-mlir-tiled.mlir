// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir{tile-sizes=16,24} | filecheck %s

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
// CHECK-NEXT: func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
// CHECK-NEXT:   %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:   %3 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:   %4 = arith.constant 1 : index
// CHECK-NEXT:   %5 = arith.constant 2 : index
// CHECK-NEXT:   %6 = arith.constant 3 : index
// CHECK-NEXT:   %7 = arith.constant 1 : index
// CHECK-NEXT:   %8 = arith.constant 65 : index
// CHECK-NEXT:   %9 = arith.constant 66 : index
// CHECK-NEXT:   %10 = arith.constant 16 : index
// CHECK-NEXT:   %11 = arith.constant 24 : index
// CHECK-NEXT:   %12 = arith.constant 63 : index
// CHECK-NEXT:   "scf.parallel"(%4, %8, %10) ({
// CHECK-NEXT:   ^0(%13 : index):
// CHECK-NEXT:     "scf.for"(%5, %9, %11) ({
// CHECK-NEXT:     ^1(%14 : index):
// CHECK-NEXT:       %15 = arith.addi %13, %10 : index
// CHECK-NEXT:       %16 = "arith.cmpi"(%15, %8) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:       %17 = "arith.select"(%16, %15, %8) : (i1, index, index) -> index
// CHECK-NEXT:       "scf.for"(%13, %17, %7) ({
// CHECK-NEXT:       ^2(%18 : index):
// CHECK-NEXT:         %19 = arith.addi %14, %11 : index
// CHECK-NEXT:         %20 = "arith.cmpi"(%19, %9) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:         %21 = "arith.select"(%20, %19, %9) : (i1, index, index) -> index
// CHECK-NEXT:         "scf.for"(%14, %21, %7) ({
// CHECK-NEXT:         ^3(%22 : index):
// CHECK-NEXT:           "scf.for"(%6, %12, %7) ({
// CHECK-NEXT:           ^4(%23 : index):
// CHECK-NEXT:             %24 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:             %25 = arith.addf %0, %24 : f64
// CHECK-NEXT:             "memref.store"(%25, %3, %18, %22, %23) : (f64, memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>, index, index, index) -> ()
// CHECK-NEXT:             "scf.yield"() : () -> ()
// CHECK-NEXT:           }) : (index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
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

// CHECK-NEXT: func.func @bufferswapping(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
// CHECK-NEXT:   %time_m = arith.constant 0 : index
// CHECK-NEXT:   %time_M = arith.constant 1001 : index
// CHECK-NEXT:   %step = arith.constant 1 : index
// CHECK-NEXT:   %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
// CHECK-NEXT:   ^5(%time : index, %fim1 : memref<2004x2004xf32>, %fi : memref<2004x2004xf32>):
// CHECK-NEXT:     %fi_storeview = "memref.subview"(%fi) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:     %fim1_loadview = "memref.subview"(%fim1) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:     %26 = arith.constant 0 : index
// CHECK-NEXT:     %27 = arith.constant 0 : index
// CHECK-NEXT:     %28 = arith.constant 1 : index
// CHECK-NEXT:     %29 = arith.constant 2000 : index
// CHECK-NEXT:     %30 = arith.constant 16 : index
// CHECK-NEXT:     %31 = arith.constant 24 : index
// CHECK-NEXT:     %32 = arith.constant 2000 : index
// CHECK-NEXT:     "scf.parallel"(%26, %29, %30) ({
// CHECK-NEXT:     ^6(%33 : index):
// CHECK-NEXT:       "scf.for"(%27, %32, %31) ({
// CHECK-NEXT:       ^7(%34 : index):
// CHECK-NEXT:         %35 = arith.addi %33, %30 : index
// CHECK-NEXT:         %36 = "arith.cmpi"(%35, %29) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:         %37 = "arith.select"(%36, %35, %29) : (i1, index, index) -> index
// CHECK-NEXT:         "scf.for"(%33, %37, %28) ({
// CHECK-NEXT:         ^8(%38 : index):
// CHECK-NEXT:           %39 = arith.addi %34, %31 : index
// CHECK-NEXT:           %40 = "arith.cmpi"(%39, %32) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:           %41 = "arith.select"(%40, %39, %32) : (i1, index, index) -> index
// CHECK-NEXT:           "scf.for"(%34, %41, %28) ({
// CHECK-NEXT:           ^9(%42 : index):
// CHECK-NEXT:             %i = "memref.load"(%fim1_loadview, %38, %42) : (memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> f32
// CHECK-NEXT:             "memref.store"(%i, %fi_storeview, %38, %42) : (f32, memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> ()
// CHECK-NEXT:             "scf.yield"() : () -> ()
// CHECK-NEXT:           }) : (index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
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

// CHECK-NEXT: func.func @copy_1d(%43 : memref<?xf64>, %out : memref<?xf64>) {
// CHECK-NEXT:   %44 = "memref.cast"(%43) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:   %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
// CHECK-NEXT:   %outc_storeview = "memref.subview"(%outc) {"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
// CHECK-NEXT:   %45 = "memref.subview"(%44) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:   %46 = arith.constant 0 : index
// CHECK-NEXT:   %47 = arith.constant 1 : index
// CHECK-NEXT:   %48 = arith.constant 16 : index
// CHECK-NEXT:   %49 = arith.constant 68 : index
// CHECK-NEXT:   "scf.parallel"(%46, %49, %48) ({
// CHECK-NEXT:   ^10(%50 : index):
// CHECK-NEXT:     %51 = arith.addi %50, %48 : index
// CHECK-NEXT:     %52 = "arith.cmpi"(%51, %49) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:     %53 = "arith.select"(%52, %51, %49) : (i1, index, index) -> index
// CHECK-NEXT:     "scf.for"(%50, %53, %47) ({
// CHECK-NEXT:     ^11(%54 : index):
// CHECK-NEXT:       %55 = arith.constant -1 : index
// CHECK-NEXT:       %56 = arith.addi %54, %55 : index
// CHECK-NEXT:       %57 = "memref.load"(%45, %56) : (memref<69xf64, strided<[1], offset: 4>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%57, %outc_storeview, %54) : (f64, memref<68xf64, strided<[1]>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
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

  // CHECK-NEXT: func.func @copy_2d(%58 : memref<?x?xf64>) {
  // CHECK-NEXT:   %59 = "memref.cast"(%58) : (memref<?x?xf64>) -> memref<72x72xf64>
  // CHECK-NEXT:   %60 = "memref.subview"(%59) {"static_offsets" = array<i64: 4, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 292>>
  // CHECK-NEXT:   %61 = arith.constant 0 : index
  // CHECK-NEXT:   %62 = arith.constant 0 : index
  // CHECK-NEXT:   %63 = arith.constant 1 : index
  // CHECK-NEXT:   %64 = arith.constant 64 : index
  // CHECK-NEXT:   %65 = arith.constant 16 : index
  // CHECK-NEXT:   %66 = arith.constant 24 : index
  // CHECK-NEXT:   %67 = arith.constant 68 : index
  // CHECK-NEXT:   "scf.parallel"(%61, %64, %65) ({
  // CHECK-NEXT:   ^12(%68 : index):
  // CHECK-NEXT:     "scf.for"(%62, %67, %66) ({
  // CHECK-NEXT:     ^13(%69 : index):
  // CHECK-NEXT:       %70 = arith.addi %68, %65 : index
  // CHECK-NEXT:       %71 = "arith.cmpi"(%70, %64) {"predicate" = 6 : i64} : (index, index) -> i1
  // CHECK-NEXT:       %72 = "arith.select"(%71, %70, %64) : (i1, index, index) -> index
  // CHECK-NEXT:       "scf.for"(%68, %72, %63) ({
  // CHECK-NEXT:       ^14(%73 : index):
  // CHECK-NEXT:         %74 = arith.addi %69, %66 : index
  // CHECK-NEXT:         %75 = "arith.cmpi"(%74, %67) {"predicate" = 6 : i64} : (index, index) -> i1
  // CHECK-NEXT:         %76 = "arith.select"(%75, %74, %67) : (i1, index, index) -> index
  // CHECK-NEXT:         "scf.for"(%69, %76, %63) ({
  // CHECK-NEXT:         ^15(%77 : index):
  // CHECK-NEXT:           %78 = arith.constant -1 : index
  // CHECK-NEXT:           %79 = arith.addi %73, %78 : index
  // CHECK-NEXT:           %80 = "memref.load"(%60, %79, %77) : (memref<65x68xf64, strided<[72, 1], offset: 292>>, index, index) -> f64
  // CHECK-NEXT:           "scf.yield"() : () -> ()
  // CHECK-NEXT:         }) : (index, index, index) -> ()
  // CHECK-NEXT:         "scf.yield"() : () -> ()
  // CHECK-NEXT:       }) : (index, index, index) -> ()
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
  
// CHECK-NEXT: func.func @copy_3d(%81 : memref<?x?x?xf64>) {
// CHECK-NEXT:   %82 = "memref.cast"(%81) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:   %83 = "memref.subview"(%82) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:   %84 = arith.constant 0 : index
// CHECK-NEXT:   %85 = arith.constant 0 : index
// CHECK-NEXT:   %86 = arith.constant 0 : index
// CHECK-NEXT:   %87 = arith.constant 1 : index
// CHECK-NEXT:   %88 = arith.constant 64 : index
// CHECK-NEXT:   %89 = arith.constant 64 : index
// CHECK-NEXT:   %90 = arith.constant 16 : index
// CHECK-NEXT:   %91 = arith.constant 24 : index
// CHECK-NEXT:   %92 = arith.constant 68 : index
// CHECK-NEXT:   "scf.parallel"(%84, %88, %90) ({
// CHECK-NEXT:   ^16(%93 : index):
// CHECK-NEXT:     "scf.for"(%85, %89, %91) ({
// CHECK-NEXT:     ^17(%94 : index):
// CHECK-NEXT:       %95 = arith.addi %93, %90 : index
// CHECK-NEXT:       %96 = "arith.cmpi"(%95, %88) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:       %97 = "arith.select"(%96, %95, %88) : (i1, index, index) -> index
// CHECK-NEXT:       "scf.for"(%93, %97, %87) ({
// CHECK-NEXT:       ^18(%98 : index):
// CHECK-NEXT:         %99 = arith.addi %94, %91 : index
// CHECK-NEXT:         %100 = "arith.cmpi"(%99, %89) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:         %101 = "arith.select"(%100, %99, %89) : (i1, index, index) -> index
// CHECK-NEXT:         "scf.for"(%94, %101, %87) ({
// CHECK-NEXT:         ^19(%102 : index):
// CHECK-NEXT:           "scf.for"(%86, %92, %87) ({
// CHECK-NEXT:           ^20(%103 : index):
// CHECK-NEXT:             %104 = arith.constant -1 : index
// CHECK-NEXT:             %105 = arith.addi %98, %104 : index
// CHECK-NEXT:             %106 = arith.constant 1 : index
// CHECK-NEXT:             %107 = arith.addi %103, %106 : index
// CHECK-NEXT:             %108 = "memref.load"(%83, %105, %102, %107) : (memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>, index, index, index) -> f64
// CHECK-NEXT:             "scf.yield"() : () -> ()
// CHECK-NEXT:           }) : (index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    func.return
  }
  // CHECK:      func.func @test_funcop_lowering(%109 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  func.func @test_funcop_lowering_dyn(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    func.return
  }
  // CHECK-NEXT: func.func @test_funcop_lowering_dyn(%110 : memref<8x8xf64>) {
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

// CHECK-NEXT: func.func @offsets(%111 : memref<?x?x?xf64>, %112 : memref<?x?x?xf64>, %113 : memref<?x?x?xf64>) {
// CHECK-NEXT:   %114 = "memref.cast"(%111) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %115 = "memref.cast"(%112) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %116 = "memref.subview"(%115) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:   %117 = "memref.cast"(%113) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:   %118 = "memref.subview"(%114) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:   %119 = arith.constant 0 : index
// CHECK-NEXT:   %120 = arith.constant 0 : index
// CHECK-NEXT:   %121 = arith.constant 0 : index
// CHECK-NEXT:   %122 = arith.constant 1 : index
// CHECK-NEXT:   %123 = arith.constant 64 : index
// CHECK-NEXT:   %124 = arith.constant 64 : index
// CHECK-NEXT:   %125 = arith.constant 16 : index
// CHECK-NEXT:   %126 = arith.constant 24 : index
// CHECK-NEXT:   %127 = arith.constant 64 : index
// CHECK-NEXT:   "scf.parallel"(%119, %123, %125) ({
// CHECK-NEXT:   ^21(%128 : index):
// CHECK-NEXT:     "scf.for"(%120, %124, %126) ({
// CHECK-NEXT:     ^22(%129 : index):
// CHECK-NEXT:       %130 = arith.addi %128, %125 : index
// CHECK-NEXT:       %131 = "arith.cmpi"(%130, %123) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:       %132 = "arith.select"(%131, %130, %123) : (i1, index, index) -> index
// CHECK-NEXT:       "scf.for"(%128, %132, %122) ({
// CHECK-NEXT:       ^23(%133 : index):
// CHECK-NEXT:         %134 = arith.addi %129, %126 : index
// CHECK-NEXT:         %135 = "arith.cmpi"(%134, %124) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:         %136 = "arith.select"(%135, %134, %124) : (i1, index, index) -> index
// CHECK-NEXT:         "scf.for"(%129, %136, %122) ({
// CHECK-NEXT:         ^24(%137 : index):
// CHECK-NEXT:           "scf.for"(%121, %127, %122) ({
// CHECK-NEXT:           ^25(%138 : index):
// CHECK-NEXT:             %139 = arith.constant -1 : index
// CHECK-NEXT:             %140 = arith.addi %133, %139 : index
// CHECK-NEXT:             %141 = "memref.load"(%118, %140, %137, %138) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:             %142 = arith.constant 1 : index
// CHECK-NEXT:             %143 = arith.addi %133, %142 : index
// CHECK-NEXT:             %144 = "memref.load"(%118, %143, %137, %138) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:             %145 = arith.constant 1 : index
// CHECK-NEXT:             %146 = arith.addi %137, %145 : index
// CHECK-NEXT:             %147 = "memref.load"(%118, %133, %146, %138) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:             %148 = arith.constant -1 : index
// CHECK-NEXT:             %149 = arith.addi %137, %148 : index
// CHECK-NEXT:             %150 = "memref.load"(%118, %133, %149, %138) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:             %151 = "memref.load"(%118, %133, %137, %138) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:             %152 = arith.addf %141, %144 : f64
// CHECK-NEXT:             %153 = arith.addf %147, %150 : f64
// CHECK-NEXT:             %154 = arith.addf %152, %153 : f64
// CHECK-NEXT:             %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:             %155 = arith.mulf %151, %cst : f64
// CHECK-NEXT:             %156 = arith.addf %155, %154 : f64
// CHECK-NEXT:             "memref.store"(%156, %116, %133, %137, %138) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
// CHECK-NEXT:             "scf.yield"() : () -> ()
// CHECK-NEXT:           }) : (index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
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

// CHECK-NEXT: func.func @neg_bounds(%in : memref<64xf64>, %out_1 : memref<64xf64>) {
// CHECK-NEXT:   %out_storeview = "memref.subview"(%out_1) {"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:   %in_loadview = "memref.subview"(%in) {"static_offsets" = array<i64: 32>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<32xf64, strided<[1], offset: 32>>
// CHECK-NEXT:   %157 = arith.constant -16 : index
// CHECK-NEXT:   %158 = arith.constant 1 : index
// CHECK-NEXT:   %159 = arith.constant 16 : index
// CHECK-NEXT:   %160 = arith.constant 16 : index
// CHECK-NEXT:   "scf.parallel"(%157, %160, %159) ({
// CHECK-NEXT:   ^26(%161 : index):
// CHECK-NEXT:     %162 = arith.addi %161, %159 : index
// CHECK-NEXT:     %163 = "arith.cmpi"(%162, %160) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:     %164 = "arith.select"(%163, %162, %160) : (i1, index, index) -> index
// CHECK-NEXT:     "scf.for"(%161, %164, %158) ({
// CHECK-NEXT:     ^27(%165 : index):
// CHECK-NEXT:       %val = "memref.load"(%in_loadview, %165) : (memref<32xf64, strided<[1], offset: 32>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%val, %out_storeview, %165) : (f64, memref<32xf64, strided<[1], offset: 32>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
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
  
// CHECK-NEXT: func.func @stencil_buffer(%166 : memref<72xf64>, %167 : memref<72xf64>) {
// CHECK-NEXT:   %168 = "memref.subview"(%167) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:   %169 = "memref.subview"(%166) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:   %170 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<64xf64>
// CHECK-NEXT:   %171 = "memref.subview"(%170) {"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<64xf64>) -> memref<64xf64, strided<[1], offset: -1>>
// CHECK-NEXT:   %172 = arith.constant 1 : index
// CHECK-NEXT:   %173 = arith.constant 1 : index
// CHECK-NEXT:   %174 = arith.constant 16 : index
// CHECK-NEXT:   %175 = arith.constant 65 : index
// CHECK-NEXT:   "scf.parallel"(%172, %175, %174) ({
// CHECK-NEXT:   ^28(%176 : index):
// CHECK-NEXT:     %177 = arith.addi %176, %174 : index
// CHECK-NEXT:     %178 = "arith.cmpi"(%177, %175) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:     %179 = "arith.select"(%178, %177, %175) : (i1, index, index) -> index
// CHECK-NEXT:     "scf.for"(%176, %179, %173) ({
// CHECK-NEXT:     ^29(%180 : index):
// CHECK-NEXT:       %181 = arith.constant -1 : index
// CHECK-NEXT:       %182 = arith.addi %180, %181 : index
// CHECK-NEXT:       %183 = "memref.load"(%169, %182) : (memref<64xf64, strided<[1], offset: 4>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%183, %171, %180) : (f64, memref<64xf64, strided<[1], offset: -1>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   %184 = arith.constant 0 : index
// CHECK-NEXT:   %185 = arith.constant 1 : index
// CHECK-NEXT:   %186 = arith.constant 16 : index
// CHECK-NEXT:   %187 = arith.constant 64 : index
// CHECK-NEXT:   "scf.parallel"(%184, %187, %186) ({
// CHECK-NEXT:   ^30(%188 : index):
// CHECK-NEXT:     %189 = arith.addi %188, %186 : index
// CHECK-NEXT:     %190 = "arith.cmpi"(%189, %187) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:     %191 = "arith.select"(%190, %189, %187) : (i1, index, index) -> index
// CHECK-NEXT:     "scf.for"(%188, %191, %185) ({
// CHECK-NEXT:     ^31(%192 : index):
// CHECK-NEXT:       %193 = arith.constant 1 : index
// CHECK-NEXT:       %194 = arith.addi %192, %193 : index
// CHECK-NEXT:       %195 = "memref.load"(%171, %194) : (memref<64xf64, strided<[1], offset: -1>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%195, %168, %192) : (f64, memref<64xf64, strided<[1], offset: 4>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   "memref.dealloc"(%171) : (memref<64xf64, strided<[1], offset: -1>>) -> ()
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

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

// CHECK-NEXT: func.func @stencil_two_stores(%196 : memref<72xf64>, %197 : memref<72xf64>, %198 : memref<72xf64>) {
// CHECK-NEXT:   %199 = "memref.subview"(%197) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:   %200 = "memref.subview"(%198) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:   %201 = "memref.subview"(%196) {"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>
// CHECK-NEXT:   %202 = arith.constant 1 : index
// CHECK-NEXT:   %203 = arith.constant 1 : index
// CHECK-NEXT:   %204 = arith.constant 16 : index
// CHECK-NEXT:   %205 = arith.constant 65 : index
// CHECK-NEXT:   "scf.parallel"(%202, %205, %204) ({
// CHECK-NEXT:   ^32(%206 : index):
// CHECK-NEXT:     %207 = arith.addi %206, %204 : index
// CHECK-NEXT:     %208 = "arith.cmpi"(%207, %205) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:     %209 = "arith.select"(%208, %207, %205) : (i1, index, index) -> index
// CHECK-NEXT:     "scf.for"(%206, %209, %203) ({
// CHECK-NEXT:     ^33(%210 : index):
// CHECK-NEXT:       %211 = arith.constant -1 : index
// CHECK-NEXT:       %212 = arith.addi %210, %211 : index
// CHECK-NEXT:       %213 = "memref.load"(%201, %212) : (memref<64xf64, strided<[1], offset: 4>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%213, %200, %210) : (f64, memref<64xf64, strided<[1], offset: 4>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   %214 = arith.constant 0 : index
// CHECK-NEXT:   %215 = arith.constant 1 : index
// CHECK-NEXT:   %216 = arith.constant 16 : index
// CHECK-NEXT:   %217 = arith.constant 64 : index
// CHECK-NEXT:   "scf.parallel"(%214, %217, %216) ({
// CHECK-NEXT:   ^34(%218 : index):
// CHECK-NEXT:     %219 = arith.addi %218, %216 : index
// CHECK-NEXT:     %220 = "arith.cmpi"(%219, %217) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:     %221 = "arith.select"(%220, %219, %217) : (i1, index, index) -> index
// CHECK-NEXT:     "scf.for"(%218, %221, %215) ({
// CHECK-NEXT:     ^35(%222 : index):
// CHECK-NEXT:       %223 = arith.constant 1 : index
// CHECK-NEXT:       %224 = arith.addi %222, %223 : index
// CHECK-NEXT:       %225 = "memref.load"(%200, %224) : (memref<64xf64, strided<[1], offset: 4>>, index) -> f64
// CHECK-NEXT:       "memref.store"(%225, %199, %222) : (f64, memref<64xf64, strided<[1], offset: 4>>, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

  func.func @apply_kernel(%69 : !stencil.field<[-2,13]x[-2,13]xf32>, %70 : !stencil.field<[-2,13]x[-2,13]xf32>, %timers : !llvm.ptr<f64>)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
    %71 = "gpu.alloc"() {"operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
    %u_vec_1 = "builtin.unrealized_conversion_cast"(%71) : (memref<15x15xf32>) -> !stencil.field<[-2,13]x[-2,13]xf32>
    %72 = "builtin.unrealized_conversion_cast"(%70) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> memref<15x15xf32>
    "gpu.memcpy"(%71, %72) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
    %73 = "gpu.alloc"() {"operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
    %u_vec_0 = "builtin.unrealized_conversion_cast"(%73) : (memref<15x15xf32>) -> !stencil.field<[-2,13]x[-2,13]xf32>
    %74 = "builtin.unrealized_conversion_cast"(%69) : (!stencil.field<[-2,13]x[-2,13]xf32>) -> memref<15x15xf32>
    "gpu.memcpy"(%73, %74) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
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

// CHECK-NEXT: func.func @apply_kernel(%226 : memref<15x15xf32>, %227 : memref<15x15xf32>, %timers : !llvm.ptr<f64>)  attributes {"param_names" = ["u_vec_0", "u_vec_1", "timers"]}{
// CHECK-NEXT:   %228 = "gpu.alloc"() {"operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
// CHECK-NEXT:   %u_vec_1 = builtin.unrealized_conversion_cast %228 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:   %229 = builtin.unrealized_conversion_cast %227 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:   "gpu.memcpy"(%228, %229) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:   %230 = "gpu.alloc"() {"operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> memref<15x15xf32>
// CHECK-NEXT:   %u_vec_0 = builtin.unrealized_conversion_cast %230 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:   %231 = builtin.unrealized_conversion_cast %226 : memref<15x15xf32> to memref<15x15xf32>
// CHECK-NEXT:   "gpu.memcpy"(%230, %231) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:   %time_m_1 = arith.constant 0 : index
// CHECK-NEXT:   %time_M_1 = arith.constant 10 : index
// CHECK-NEXT:   %step_1 = arith.constant 1 : index
// CHECK-NEXT:   %232, %233 = "scf.for"(%time_m_1, %time_M_1, %step_1, %u_vec_0, %u_vec_1) ({
// CHECK-NEXT:   ^36(%time_1 : index, %t0 : memref<15x15xf32>, %t1 : memref<15x15xf32>):
// CHECK-NEXT:     %t1_storeview = "memref.subview"(%t1) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<15x15xf32>) -> memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:     %t0_loadview = "memref.subview"(%t0) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 11, 11>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<15x15xf32>) -> memref<11x11xf32, strided<[15, 1], offset: 32>>
// CHECK-NEXT:     %234 = arith.constant 0 : index
// CHECK-NEXT:     %235 = arith.constant 0 : index
// CHECK-NEXT:     %236 = arith.constant 1 : index
// CHECK-NEXT:     %237 = arith.constant 11 : index
// CHECK-NEXT:     %238 = arith.constant 16 : index
// CHECK-NEXT:     %239 = arith.constant 24 : index
// CHECK-NEXT:     %240 = arith.constant 11 : index
// CHECK-NEXT:     "scf.parallel"(%234, %237, %238) ({
// CHECK-NEXT:     ^37(%241 : index):
// CHECK-NEXT:       "scf.for"(%235, %240, %239) ({
// CHECK-NEXT:       ^38(%242 : index):
// CHECK-NEXT:         %243 = arith.addi %241, %238 : index
// CHECK-NEXT:         %244 = "arith.cmpi"(%243, %237) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:         %245 = "arith.select"(%244, %243, %237) : (i1, index, index) -> index
// CHECK-NEXT:         "scf.for"(%241, %245, %236) ({
// CHECK-NEXT:         ^39(%246 : index):
// CHECK-NEXT:           %247 = arith.addi %242, %239 : index
// CHECK-NEXT:           %248 = "arith.cmpi"(%247, %240) {"predicate" = 6 : i64} : (index, index) -> i1
// CHECK-NEXT:           %249 = "arith.select"(%248, %247, %240) : (i1, index, index) -> index
// CHECK-NEXT:           "scf.for"(%242, %249, %236) ({
// CHECK-NEXT:           ^40(%250 : index):
// CHECK-NEXT:             %251 = "memref.load"(%t0_loadview, %246, %250) : (memref<11x11xf32, strided<[15, 1], offset: 32>>, index, index) -> f32
// CHECK-NEXT:             "memref.store"(%251, %t1_storeview, %246, %250) : (f32, memref<11x11xf32, strided<[15, 1], offset: 32>>, index, index) -> ()
// CHECK-NEXT:             "scf.yield"() : () -> ()
// CHECK-NEXT:           }) : (index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "scf.yield"(%t1, %t0) : (memref<15x15xf32>, memref<15x15xf32>) -> ()
// CHECK-NEXT:   }) : (index, index, index, memref<15x15xf32>, memref<15x15xf32>) -> (memref<15x15xf32>, memref<15x15xf32>)
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

}
// CHECK-NEXT: }
