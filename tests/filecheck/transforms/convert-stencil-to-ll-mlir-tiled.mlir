// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir{tile-sizes=16,24} | filecheck %s

builtin.module {
// CHECK: builtin.module {

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
// CHECK:    func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
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
// CHECK-NEXT:      %13 = arith.constant 16 : index
// CHECK-NEXT:      %14 = arith.constant 24 : index
// CHECK-NEXT:      %15 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%4, %11, %13) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^0(%16 : index):
// CHECK-NEXT:        scf.for %17 = %5 to %12 step %14 {
// CHECK-NEXT:          %18 = arith.addi %16, %13 : index
// CHECK-NEXT:          %19 = arith.cmpi ult, %18, %11 : index
// CHECK-NEXT:          %20 = arith.select %19, %18, %11 : index
// CHECK-NEXT:          scf.for %21 = %16 to %20 step %8 {
// CHECK-NEXT:            %22 = arith.addi %17, %14 : index
// CHECK-NEXT:            %23 = arith.cmpi ult, %22, %12 : index
// CHECK-NEXT:            %24 = arith.select %23, %22, %12 : index
// CHECK-NEXT:            scf.for %25 = %17 to %24 step %9 {
// CHECK-NEXT:              scf.for %26 = %6 to %15 step %10 {
// CHECK-NEXT:                %27 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:                %28 = arith.addf %0, %27 : f64
// CHECK-NEXT:                memref.store %28, %3[%21, %25, %26] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
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

// CHECK:         func.func @copy_1d(%29 : memref<?xf64>, %out : memref<?xf64>) {
// CHECK-NEXT:      %30 = "memref.cast"(%29) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:      %outc = "memref.cast"(%out) : (memref<?xf64>) -> memref<1024xf64>
// CHECK-NEXT:      %outc_storeview = "memref.subview"(%outc) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 68>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<1024xf64>) -> memref<68xf64, strided<[1]>>
// CHECK-NEXT:      %31 = "memref.subview"(%30) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 69>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:      %32 = arith.constant 0 : index
// CHECK-NEXT:      %33 = arith.constant 1 : index
// CHECK-NEXT:      %34 = arith.constant 1 : index
// CHECK-NEXT:      %35 = arith.constant 16 : index
// CHECK-NEXT:      %36 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%32, %36, %35) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^1(%37 : index):
// CHECK-NEXT:        %38 = arith.addi %37, %35 : index
// CHECK-NEXT:        %39 = arith.cmpi ult, %38, %36 : index
// CHECK-NEXT:        %40 = arith.select %39, %38, %36 : index
// CHECK-NEXT:        scf.for %41 = %37 to %40 step %34 {
// CHECK-NEXT:          %42 = arith.constant -1 : index
// CHECK-NEXT:          %43 = arith.addi %41, %42 : index
// CHECK-NEXT:          %44 = memref.load %31[%43] : memref<69xf64, strided<[1], offset: 4>>
// CHECK-NEXT:          memref.store %44, %outc_storeview[%41] : memref<68xf64, strided<[1]>>
// CHECK-NEXT:        }
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

// CHECK:         func.func @copy_2d(%45 : memref<?x?xf64>) {
// CHECK-NEXT:      %46 = "memref.cast"(%45) : (memref<?x?xf64>) -> memref<72x72xf64>
// CHECK-NEXT:      %47 = "memref.subview"(%46) <{"static_offsets" = array<i64: 4, 4>, "static_sizes" = array<i64: 65, 68>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72xf64>) -> memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:      %48 = arith.constant 0 : index
// CHECK-NEXT:      %49 = arith.constant 0 : index
// CHECK-NEXT:      %50 = arith.constant 1 : index
// CHECK-NEXT:      %51 = arith.constant 1 : index
// CHECK-NEXT:      %52 = arith.constant 1 : index
// CHECK-NEXT:      %53 = arith.constant 64 : index
// CHECK-NEXT:      %54 = arith.constant 16 : index
// CHECK-NEXT:      %55 = arith.constant 24 : index
// CHECK-NEXT:      %56 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%48, %53, %54) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^2(%57 : index):
// CHECK-NEXT:        scf.for %58 = %49 to %56 step %55 {
// CHECK-NEXT:          %59 = arith.addi %57, %54 : index
// CHECK-NEXT:          %60 = arith.cmpi ult, %59, %53 : index
// CHECK-NEXT:          %61 = arith.select %60, %59, %53 : index
// CHECK-NEXT:          scf.for %62 = %57 to %61 step %51 {
// CHECK-NEXT:            %63 = arith.addi %58, %55 : index
// CHECK-NEXT:            %64 = arith.cmpi ult, %63, %56 : index
// CHECK-NEXT:            %65 = arith.select %64, %63, %56 : index
// CHECK-NEXT:            scf.for %66 = %58 to %65 step %52 {
// CHECK-NEXT:              %67 = arith.constant -1 : index
// CHECK-NEXT:              %68 = arith.addi %62, %67 : index
// CHECK-NEXT:              %69 = memref.load %47[%68, %66] : memref<65x68xf64, strided<[72, 1], offset: 292>>
// CHECK-NEXT:            }
// CHECK-NEXT:          }
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

// CHECK:         func.func @copy_3d(%70 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %71 = "memref.cast"(%70) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:      %72 = "memref.subview"(%71) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 65, 64, 69>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x74x76xf64>) -> memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:      %73 = arith.constant 0 : index
// CHECK-NEXT:      %74 = arith.constant 0 : index
// CHECK-NEXT:      %75 = arith.constant 0 : index
// CHECK-NEXT:      %76 = arith.constant 1 : index
// CHECK-NEXT:      %77 = arith.constant 1 : index
// CHECK-NEXT:      %78 = arith.constant 1 : index
// CHECK-NEXT:      %79 = arith.constant 1 : index
// CHECK-NEXT:      %80 = arith.constant 64 : index
// CHECK-NEXT:      %81 = arith.constant 64 : index
// CHECK-NEXT:      %82 = arith.constant 16 : index
// CHECK-NEXT:      %83 = arith.constant 24 : index
// CHECK-NEXT:      %84 = arith.constant 68 : index
// CHECK-NEXT:      "scf.parallel"(%73, %80, %82) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^3(%85 : index):
// CHECK-NEXT:        scf.for %86 = %74 to %81 step %83 {
// CHECK-NEXT:          %87 = arith.addi %85, %82 : index
// CHECK-NEXT:          %88 = arith.cmpi ult, %87, %80 : index
// CHECK-NEXT:          %89 = arith.select %88, %87, %80 : index
// CHECK-NEXT:          scf.for %90 = %85 to %89 step %77 {
// CHECK-NEXT:            %91 = arith.addi %86, %83 : index
// CHECK-NEXT:            %92 = arith.cmpi ult, %91, %81 : index
// CHECK-NEXT:            %93 = arith.select %92, %91, %81 : index
// CHECK-NEXT:            scf.for %94 = %86 to %93 step %78 {
// CHECK-NEXT:              scf.for %95 = %75 to %84 step %79 {
// CHECK-NEXT:                %96 = arith.constant -1 : index
// CHECK-NEXT:                %97 = arith.addi %90, %96 : index
// CHECK-NEXT:                %98 = arith.constant 1 : index
// CHECK-NEXT:                %99 = arith.addi %95, %98 : index
// CHECK-NEXT:                %100 = memref.load %72[%97, %94, %99] : memref<65x64x69xf64, strided<[5624, 76, 1], offset: 22804>>
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
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

// CHECK:         func.func @offsets(%101 : memref<?x?x?xf64>, %102 : memref<?x?x?xf64>, %103 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %104 = "memref.cast"(%101) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %105 = "memref.cast"(%102) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %106 = "memref.subview"(%105) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %107 = "memref.cast"(%103) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:      %108 = "memref.subview"(%104) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:      %109 = arith.constant 0 : index
// CHECK-NEXT:      %110 = arith.constant 0 : index
// CHECK-NEXT:      %111 = arith.constant 0 : index
// CHECK-NEXT:      %112 = arith.constant 1 : index
// CHECK-NEXT:      %113 = arith.constant 1 : index
// CHECK-NEXT:      %114 = arith.constant 1 : index
// CHECK-NEXT:      %115 = arith.constant 1 : index
// CHECK-NEXT:      %116 = arith.constant 64 : index
// CHECK-NEXT:      %117 = arith.constant 64 : index
// CHECK-NEXT:      %118 = arith.constant 16 : index
// CHECK-NEXT:      %119 = arith.constant 24 : index
// CHECK-NEXT:      %120 = arith.constant 64 : index
// CHECK-NEXT:      "scf.parallel"(%109, %116, %118) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^4(%121 : index):
// CHECK-NEXT:        scf.for %122 = %110 to %117 step %119 {
// CHECK-NEXT:          %123 = arith.addi %121, %118 : index
// CHECK-NEXT:          %124 = arith.cmpi ult, %123, %116 : index
// CHECK-NEXT:          %125 = arith.select %124, %123, %116 : index
// CHECK-NEXT:          scf.for %126 = %121 to %125 step %113 {
// CHECK-NEXT:            %127 = arith.addi %122, %119 : index
// CHECK-NEXT:            %128 = arith.cmpi ult, %127, %117 : index
// CHECK-NEXT:            %129 = arith.select %128, %127, %117 : index
// CHECK-NEXT:            scf.for %130 = %122 to %129 step %114 {
// CHECK-NEXT:              scf.for %131 = %111 to %120 step %115 {
// CHECK-NEXT:                %132 = arith.constant -1 : index
// CHECK-NEXT:                %133 = arith.addi %126, %132 : index
// CHECK-NEXT:                %134 = memref.load %108[%133, %130, %131] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:                %135 = arith.constant 1 : index
// CHECK-NEXT:                %136 = arith.addi %126, %135 : index
// CHECK-NEXT:                %137 = memref.load %108[%136, %130, %131] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:                %138 = arith.constant 1 : index
// CHECK-NEXT:                %139 = arith.addi %130, %138 : index
// CHECK-NEXT:                %140 = memref.load %108[%126, %139, %131] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:                %141 = arith.constant -1 : index
// CHECK-NEXT:                %142 = arith.addi %130, %141 : index
// CHECK-NEXT:                %143 = memref.load %108[%126, %142, %131] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:                %144 = memref.load %108[%126, %130, %131] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:                %145 = arith.addf %134, %137 : f64
// CHECK-NEXT:                %146 = arith.addf %140, %143 : f64
// CHECK-NEXT:                %147 = arith.addf %145, %146 : f64
// CHECK-NEXT:                %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:                %148 = arith.mulf %144, %cst : f64
// CHECK-NEXT:                %149 = arith.addf %148, %147 : f64
// CHECK-NEXT:                memref.store %149, %106[%126, %130, %131] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @stencil_init_float_unrolled(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = arith.constant 1.0 : f64
      %6 = arith.addf %4, %5 : f64
      "stencil.return"(%6, %6, %6, %6, %6, %6, %6, %6) <{"unroll" = #stencil.index<2, 2, 2>}> : (f64, f64, f64, f64, f64, f64, f64, f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
    func.return

  }

// CHECK:         func.func @stencil_init_float_unrolled(%150 : f64, %151 : memref<?x?x?xf64>) {
// CHECK-NEXT:      %152 = "memref.cast"(%151) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:      %153 = "memref.subview"(%152) <{"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:      %154 = arith.constant 1 : index
// CHECK-NEXT:      %155 = arith.constant 2 : index
// CHECK-NEXT:      %156 = arith.constant 3 : index
// CHECK-NEXT:      %157 = arith.constant 1 : index
// CHECK-NEXT:      %158 = arith.constant 2 : index
// CHECK-NEXT:      %159 = arith.constant 2 : index
// CHECK-NEXT:      %160 = arith.constant 2 : index
// CHECK-NEXT:      %161 = arith.constant 65 : index
// CHECK-NEXT:      %162 = arith.constant 66 : index
// CHECK-NEXT:      %163 = arith.constant 16 : index
// CHECK-NEXT:      %164 = arith.constant 24 : index
// CHECK-NEXT:      %165 = arith.constant 63 : index
// CHECK-NEXT:      "scf.parallel"(%154, %161, %163) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^5(%166 : index):
// CHECK-NEXT:        scf.for %167 = %155 to %162 step %164 {
// CHECK-NEXT:          %168 = arith.addi %166, %163 : index
// CHECK-NEXT:          %169 = arith.cmpi ult, %168, %161 : index
// CHECK-NEXT:          %170 = arith.select %169, %168, %161 : index
// CHECK-NEXT:          scf.for %171 = %166 to %170 step %158 {
// CHECK-NEXT:            %172 = arith.addi %167, %164 : index
// CHECK-NEXT:            %173 = arith.cmpi ult, %172, %162 : index
// CHECK-NEXT:            %174 = arith.select %173, %172, %162 : index
// CHECK-NEXT:            scf.for %175 = %167 to %174 step %159 {
// CHECK-NEXT:              scf.for %176 = %156 to %165 step %160 {
// CHECK-NEXT:                %177 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:                %178 = arith.addf %150, %177 : f64
// CHECK-NEXT:                memref.store %178, %153[%171, %175, %176] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %179 = arith.constant 1 : index
// CHECK-NEXT:                %180 = arith.addi %176, %179 : index
// CHECK-NEXT:                memref.store %178, %153[%171, %175, %180] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %181 = arith.constant 1 : index
// CHECK-NEXT:                %182 = arith.addi %175, %181 : index
// CHECK-NEXT:                memref.store %178, %153[%171, %182, %176] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %183 = arith.constant 1 : index
// CHECK-NEXT:                %184 = arith.addi %175, %183 : index
// CHECK-NEXT:                %185 = arith.constant 1 : index
// CHECK-NEXT:                %186 = arith.addi %176, %185 : index
// CHECK-NEXT:                memref.store %178, %153[%171, %184, %186] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %187 = arith.constant 1 : index
// CHECK-NEXT:                %188 = arith.addi %171, %187 : index
// CHECK-NEXT:                memref.store %178, %153[%188, %175, %176] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %189 = arith.constant 1 : index
// CHECK-NEXT:                %190 = arith.addi %171, %189 : index
// CHECK-NEXT:                %191 = arith.constant 1 : index
// CHECK-NEXT:                %192 = arith.addi %176, %191 : index
// CHECK-NEXT:                memref.store %178, %153[%190, %175, %192] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %193 = arith.constant 1 : index
// CHECK-NEXT:                %194 = arith.addi %171, %193 : index
// CHECK-NEXT:                %195 = arith.constant 1 : index
// CHECK-NEXT:                %196 = arith.addi %175, %195 : index
// CHECK-NEXT:                memref.store %178, %153[%194, %196, %176] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:                %197 = arith.constant 1 : index
// CHECK-NEXT:                %198 = arith.addi %171, %197 : index
// CHECK-NEXT:                %199 = arith.constant 1 : index
// CHECK-NEXT:                %200 = arith.addi %175, %199 : index
// CHECK-NEXT:                %201 = arith.constant 1 : index
// CHECK-NEXT:                %202 = arith.addi %176, %201 : index
// CHECK-NEXT:                memref.store %178, %153[%198, %200, %202] : memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}
// CHECK-NEXT: }
