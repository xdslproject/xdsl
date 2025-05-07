// RUN: xdsl-opt %s -p "distribute-stencil{strategy=3d-grid slices=2,2,2}" | filecheck %s
// RUN: xdsl-opt %s -p "distribute-stencil{strategy=3d-grid slices=2,2,2},shape-inference" | filecheck %s --check-prefix SHAPE
// RUN: xdsl-opt %s -p "distribute-stencil{strategy=3d-grid slices=2,2,2},shape-inference,stencil-bufferize" | filecheck %s --check-prefix BUFF

  func.func @offsets(%27 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %28 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %29 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
    %33 = stencil.load %27 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
    %34, %35 = stencil.apply(%36 = %33 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %37 = stencil.access %36[-1, 0, 0] : !stencil.temp<?x?x?xf64>
      %38 = stencil.access %36[1, 0, 0] : !stencil.temp<?x?x?xf64>
      %39 = stencil.access %36[0, 1, 0] : !stencil.temp<?x?x?xf64>
      %40 = stencil.access %36[0, -1, 0] : !stencil.temp<?x?x?xf64>
      %41 = stencil.access %36[0, 0, 0] : !stencil.temp<?x?x?xf64>
      %42 = arith.addf %37, %38 : f64
      %43 = arith.addf %39, %40 : f64
      %44 = arith.addf %42, %43 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %45 = arith.mulf %41, %cst : f64
      %46 = arith.addf %45, %44 : f64
      stencil.return %46, %45 : f64, f64
    }
    stencil.store %34 to %28(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    stencil.store %35 to %29(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @offsets(%0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// CHECK-NEXT:      %3 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %4 = "dmp.swap"(%3) {strategy = #dmp.grid_slice_3d<#dmp.topo<2x2x2>, false>, swaps = []} : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %5, %6 = stencil.apply(%7 = %4 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %8 = stencil.access %7[-1, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %9 = stencil.access %7[1, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %10 = stencil.access %7[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %11 = stencil.access %7[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %12 = stencil.access %7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %13 = arith.addf %8, %9 : f64
// CHECK-NEXT:        %14 = arith.addf %10, %11 : f64
// CHECK-NEXT:        %15 = arith.addf %13, %14 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %16 = arith.mulf %12, %cst : f64
// CHECK-NEXT:        %17 = arith.addf %16, %15 : f64
// CHECK-NEXT:        stencil.return %17, %16 : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %5 to %1(<[0, 0, 0], [32, 32, 32]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      stencil.store %6 to %2(<[0, 0, 0], [32, 32, 32]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// SHAPE:         func.func @offsets(%0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// SHAPE-NEXT:      %3 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:      %4 = "dmp.swap"(%3) {strategy = #dmp.grid_slice_3d<#dmp.topo<2x2x2>, false>, swaps = [#dmp.exchange<at [32, 0, 0] size [1, 32, 32] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 32, 32] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 32, 0] size [32, 1, 32] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [32, 1, 32] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>) -> !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:      %5, %6 = stencil.apply(%7 = %4 : !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>) -> (!stencil.temp<[0,32]x[0,32]x[0,32]xf64>, !stencil.temp<[0,32]x[0,32]x[0,32]xf64>) {
// SHAPE-NEXT:        %8 = stencil.access %7[-1, 0, 0] : !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:        %9 = stencil.access %7[1, 0, 0] : !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:        %10 = stencil.access %7[0, 1, 0] : !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:        %11 = stencil.access %7[0, -1, 0] : !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:        %12 = stencil.access %7[0, 0, 0] : !stencil.temp<[-1,33]x[-1,33]x[0,32]xf64>
// SHAPE-NEXT:        %13 = arith.addf %8, %9 : f64
// SHAPE-NEXT:        %14 = arith.addf %10, %11 : f64
// SHAPE-NEXT:        %15 = arith.addf %13, %14 : f64
// SHAPE-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// SHAPE-NEXT:        %16 = arith.mulf %12, %cst : f64
// SHAPE-NEXT:        %17 = arith.addf %16, %15 : f64
// SHAPE-NEXT:        stencil.return %17, %16 : f64, f64
// SHAPE-NEXT:      }
// SHAPE-NEXT:      stencil.store %5 to %1(<[0, 0, 0], [32, 32, 32]>) : !stencil.temp<[0,32]x[0,32]x[0,32]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      stencil.store %6 to %2(<[0, 0, 0], [32, 32, 32]>) : !stencil.temp<[0,32]x[0,32]x[0,32]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }

// BUFF:         func.func @offsets(%0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// BUFF-NEXT:      "dmp.swap"(%0) {strategy = #dmp.grid_slice_3d<#dmp.topo<2x2x2>, false>, swaps = [#dmp.exchange<at [32, 0, 0] size [1, 32, 32] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 32, 32] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 32, 0] size [32, 1, 32] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [32, 1, 32] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// BUFF-NEXT:      stencil.apply(%3 = %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// BUFF-NEXT:        %4 = stencil.access %3[-1, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %5 = stencil.access %3[1, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %6 = stencil.access %3[0, 1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %7 = stencil.access %3[0, -1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %8 = stencil.access %3[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %9 = arith.addf %4, %5 : f64
// BUFF-NEXT:        %10 = arith.addf %6, %7 : f64
// BUFF-NEXT:        %11 = arith.addf %9, %10 : f64
// BUFF-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// BUFF-NEXT:        %12 = arith.mulf %8, %cst : f64
// BUFF-NEXT:        %13 = arith.addf %12, %11 : f64
// BUFF-NEXT:        stencil.return %13, %12 : f64, f64
// BUFF-NEXT:      } to <[0, 0, 0], [32, 32, 32]>
// BUFF-NEXT:      func.return
// BUFF-NEXT:    }

  func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
    stencil.external_store %dyn_field to %dyn_mem : !stencil.field<?x?x?xf64> to memref<?x?x?xf64>
    stencil.external_store %sta_field to %sta_mem : !stencil.field<[-2,62]x[0,64]x[2,66]xf64> to memref<64x64x64xf64>
    %47 = stencil.external_load %dyn_mem : memref<?x?x?xf64> -> !stencil.field<?x?x?xf64>
    %48 = stencil.external_load %sta_mem : memref<64x64x64xf64> -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
    %casted = stencil.cast %47 : !stencil.field<?x?x?xf64> -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
    func.return
  }

// SHAPE:         func.func @trivial_externals(%dyn_mem : memref<?x?x?xf64>, %sta_mem : memref<64x64x64xf64>, %dyn_field : !stencil.field<?x?x?xf64>, %sta_field : !stencil.field<[-2,62]x[0,64]x[2,66]xf64>) {
// SHAPE-NEXT:      stencil.external_store %dyn_field to %dyn_mem : !stencil.field<?x?x?xf64> to memref<?x?x?xf64>
// SHAPE-NEXT:      stencil.external_store %sta_field to %sta_mem : !stencil.field<[-2,62]x[0,64]x[2,66]xf64> to memref<64x64x64xf64>
// SHAPE-NEXT:      %0 = stencil.external_load %dyn_mem : memref<?x?x?xf64> -> !stencil.field<?x?x?xf64>
// SHAPE-NEXT:      %1 = stencil.external_load %sta_mem : memref<64x64x64xf64> -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
// SHAPE-NEXT:      %casted = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-2,62]x[0,64]x[2,66]xf64>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }

  func.func @stencil_init_index(%91 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %92 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x = stencil.index 0 <[0, 0, 0]>
      %y = stencil.index 1 <[0, 0, 0]>
      %z = stencil.index 2 <[0, 0, 0]>
      %xy = arith.addi %x, %y : index
      %xyz = arith.addi %xy, %z : index
      stencil.return %xyz : index
    }
    stencil.store %92 to %91(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// SHAPE:         func.func @stencil_init_index(%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
// SHAPE-NEXT:      %1 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
// SHAPE-NEXT:        %x = stencil.index 0 <[0, 0, 0]>
// SHAPE-NEXT:        %y = stencil.index 1 <[0, 0, 0]>
// SHAPE-NEXT:        %z = stencil.index 2 <[0, 0, 0]>
// SHAPE-NEXT:        %xy = arith.addi %x, %y : index
// SHAPE-NEXT:        %xyz = arith.addi %xy, %z : index
// SHAPE-NEXT:        stencil.return %xyz : index
// SHAPE-NEXT:      }
// SHAPE-NEXT:      stencil.store %1 to %0(<[0, 0, 0], [32, 32, 32]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }

  func.func @stencil_init_index_offset(%93 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
    %94 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
      %x_1 = stencil.index 0 <[1, 1, 1]>
      %y_1 = stencil.index 1 <[-1, -1, -1]>
      %z_1 = stencil.index 2 <[0, 0, 0]>
      %xy_1 = arith.addi %x_1, %y_1 : index
      %xyz_1 = arith.addi %xy_1, %z_1 : index
      stencil.return %xyz_1 : index
    }
    stencil.store %94 to %93(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
    func.return
  }

// SHAPE:         func.func @stencil_init_index_offset(%0 : !stencil.field<[0,64]x[0,64]x[0,64]xindex>) {
// SHAPE-NEXT:      %1 = stencil.apply() -> (!stencil.temp<[0,64]x[0,64]x[0,64]xindex>) {
// SHAPE-NEXT:        %x = stencil.index 0 <[1, 1, 1]>
// SHAPE-NEXT:        %y = stencil.index 1 <[-1, -1, -1]>
// SHAPE-NEXT:        %z = stencil.index 2 <[0, 0, 0]>
// SHAPE-NEXT:        %xy = arith.addi %x, %y : index
// SHAPE-NEXT:        %xyz = arith.addi %xy, %z : index
// SHAPE-NEXT:        stencil.return %xyz : index
// SHAPE-NEXT:      }
// SHAPE-NEXT:      stencil.store %1 to %0(<[0, 0, 0], [32, 32, 32]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xindex> to !stencil.field<[0,64]x[0,64]x[0,64]xindex>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }

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

// SHAPE:         func.func @store_result_lowering(%arg0 : f64) {
// SHAPE-NEXT:      %0, %1 = stencil.apply(%arg1 = %arg0 : f64) -> (!stencil.temp<[0,7]x[0,7]x[0,7]xf64>, !stencil.temp<[0,7]x[0,7]x[0,7]xf64>) {
// SHAPE-NEXT:        %2 = stencil.store_result %arg1 : !stencil.result<f64>
// SHAPE-NEXT:        %3 = stencil.store_result %arg1 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %2, %3 : !stencil.result<f64>, !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %2 = stencil.buffer %1 : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> -> !stencil.temp<[0,7]x[0,7]x[0,7]xf64>
// SHAPE-NEXT:      %3 = stencil.buffer %0 : !stencil.temp<[0,7]x[0,7]x[0,7]xf64> -> !stencil.temp<[0,7]x[0,7]x[0,7]xf64>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }

func.func @if_lowering(%arg0_1 : f64, %b0 : !stencil.field<[0,8]x[0,8]x[0,8]xf64>, %b1 : !stencil.field<[0,8]x[0,8]x[0,8]xf64>)  attributes {"stencil.program"} {
    %101, %102 = stencil.apply(%arg1_1 = %arg0_1 : f64) -> (!stencil.temp<[0,8]x[0,8]x[0,8]xf64>, !stencil.temp<[0,8]x[0,8]x[0,8]xf64>) {
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
    stencil.store %101 to %b0(<[0, 0, 0], [8, 8, 8]>) : !stencil.temp<[0,8]x[0,8]x[0,8]xf64> to !stencil.field<[0,8]x[0,8]x[0,8]xf64>
    stencil.store %102 to %b1(<[0, 0, 0], [8, 8, 8]>) : !stencil.temp<[0,8]x[0,8]x[0,8]xf64> to !stencil.field<[0,8]x[0,8]x[0,8]xf64>
    func.return
  }

// SHAPE:         func.func @if_lowering(%arg0 : f64, %b0 : !stencil.field<[0,8]x[0,8]x[0,8]xf64>, %b1 : !stencil.field<[0,8]x[0,8]x[0,8]xf64>)  attributes {stencil.program} {
// SHAPE-NEXT:      %0, %1 = stencil.apply(%arg1 = %arg0 : f64) -> (!stencil.temp<[0,8]x[0,8]x[0,8]xf64>, !stencil.temp<[0,8]x[0,8]x[0,8]xf64>) {
// SHAPE-NEXT:        %true = "test.op"() : () -> i1
// SHAPE-NEXT:        %2, %3 = scf.if %true -> (!stencil.result<f64>, f64) {
// SHAPE-NEXT:          %4 = stencil.store_result %arg1 : !stencil.result<f64>
// SHAPE-NEXT:          scf.yield %4, %arg1 : !stencil.result<f64>, f64
// SHAPE-NEXT:        } else {
// SHAPE-NEXT:          %5 = stencil.store_result  : !stencil.result<f64>
// SHAPE-NEXT:          scf.yield %5, %arg1 : !stencil.result<f64>, f64
// SHAPE-NEXT:        }
// SHAPE-NEXT:        %6 = stencil.store_result %3 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %2, %6 : !stencil.result<f64>, !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      stencil.store %0 to %b0(<[0, 0, 0], [4, 4, 4]>) : !stencil.temp<[0,8]x[0,8]x[0,8]xf64> to !stencil.field<[0,8]x[0,8]x[0,8]xf64>
// SHAPE-NEXT:      stencil.store %1 to %b1(<[0, 0, 0], [4, 4, 4]>) : !stencil.temp<[0,8]x[0,8]x[0,8]xf64> to !stencil.field<[0,8]x[0,8]x[0,8]xf64>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }
