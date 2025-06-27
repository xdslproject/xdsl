// RUN: xdsl-opt -p stencil-shape-minimize --split-input-file %s | filecheck %s
// RUN: xdsl-opt -p stencil-shape-minimize{restrict=32} --split-input-file --verify-diagnostics %s | filecheck %s --check-prefix RESTRICT

builtin.module {
  func.func @different_input_offsets(%out : !stencil.field<[-4,68]xf64>, %in : !stencil.field<[-4,68]xf64>) {
    %tin = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,65]xf64>
    %tout = stencil.apply(%arg = %tin : !stencil.temp<[-1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
      %x = stencil.access %arg[-1] : !stencil.temp<[-1,65]xf64>
      %y = stencil.access %arg[1] : !stencil.temp<[-1,65]xf64>
      %o = arith.addf %x, %y : f64
      stencil.return %o : f64
    }
    stencil.store %tout to %out(<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }
}

// CHECK:       func.func @different_input_offsets(%out : !stencil.field<[-1,65]xf64>, %in : !stencil.field<[-1,65]xf64>) {
// CHECK-NEXT:    %tin = stencil.load %in : !stencil.field<[-1,65]xf64> -> !stencil.temp<[-1,65]xf64>
// CHECK-NEXT:    %tout = stencil.apply(%arg = %tin : !stencil.temp<[-1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
// CHECK-NEXT:      %x = stencil.access %arg[-1] : !stencil.temp<[-1,65]xf64>
// CHECK-NEXT:      %y = stencil.access %arg[1] : !stencil.temp<[-1,65]xf64>
// CHECK-NEXT:      %o = arith.addf %x, %y : f64
// CHECK-NEXT:      stencil.return %o : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    stencil.store %tout to %out(<[0], [64]>) : !stencil.temp<[0,64]xf64> to !stencil.field<[-1,65]xf64>
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

// RESTRICT:       func.func @different_input_offsets(%out : !stencil.field<[-1,33]xf64>, %in : !stencil.field<[-1,33]xf64>) {
// RESTRICT-NEXT:    %tin = stencil.load %in : !stencil.field<[-1,33]xf64> -> !stencil.temp<[-1,33]xf64>
// RESTRICT-NEXT:    %tout = stencil.apply(%arg = %tin : !stencil.temp<[-1,33]xf64>) -> (!stencil.temp<[0,32]xf64>) {
// RESTRICT-NEXT:      %x = stencil.access %arg[-1] : !stencil.temp<[-1,33]xf64>
// RESTRICT-NEXT:      %y = stencil.access %arg[1] : !stencil.temp<[-1,33]xf64>
// RESTRICT-NEXT:      %o = arith.addf %x, %y : f64
// RESTRICT-NEXT:      stencil.return %o : f64
// RESTRICT-NEXT:    }
// RESTRICT-NEXT:    stencil.store %tout to %out(<[0], [32]>) : !stencil.temp<[0,32]xf64> to !stencil.field<[-1,33]xf64>
// RESTRICT-NEXT:    func.return
// RESTRICT-NEXT:  }
// -----

func.func @stencil_missing_dims(%in : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %ybuf : !stencil.field<[-4,68]xf64>, %zbuf : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
  %intemp = stencil.load %in : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
  %0 = "dmp.swap"(%intemp) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>) -> !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
  %ybuf_t = stencil.load %ybuf : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,0]xf64>
  %1 = "dmp.swap"(%ybuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,0]xf64>) -> !stencil.temp<[-1,0]xf64>
  %zbuf_t = stencil.load %zbuf : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,63]xf64>
  %2 = "dmp.swap"(%zbuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = []} : (!stencil.temp<[-1,63]xf64>) -> !stencil.temp<[-1,63]xf64>
  %res = stencil.apply(%inarg = %0 : !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>, %ybuf_arg = %1 : !stencil.temp<[-1,0]xf64>, %zbuf_arg = %2 : !stencil.temp<[-1,63]xf64>) -> (!stencil.temp<[0,1]x[0,1]x[0,64]xf64>) {
    %3 = stencil.access %ybuf_arg[_, -1, _] : !stencil.temp<[-1,0]xf64>
    %4 = stencil.access %zbuf_arg[_, _, -1] : !stencil.temp<[-1,63]xf64>
    %5 = stencil.access %inarg[0, -1, -1] : !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
    %6 = arith.addf %3, %4 : f64
    %7 = arith.addf %5, %6 : f64
    stencil.return %7 : f64
  }
  stencil.store %res to %out(<[0, 0, 0], [1, 1, 64]>) : !stencil.temp<[0,1]x[0,1]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  func.return
}

// CHECK:      func.func @stencil_missing_dims(%in : !stencil.field<[0,1]x[-1,1]x[-1,64]xf64>, %ybuf : !stencil.field<[-1,63]xf64>, %zbuf : !stencil.field<[-1,63]xf64>, %out : !stencil.field<[0,1]x[-1,1]x[-1,64]xf64>) {
// CHECK-NEXT:   %intemp = stencil.load %in : !stencil.field<[0,1]x[-1,1]x[-1,64]xf64> -> !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
// CHECK-NEXT:   %0 = "dmp.swap"(%intemp) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>) -> !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
// CHECK-NEXT:   %ybuf_t = stencil.load %ybuf : !stencil.field<[-1,63]xf64> -> !stencil.temp<[-1,0]xf64>
// CHECK-NEXT:   %1 = "dmp.swap"(%ybuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,0]xf64>) -> !stencil.temp<[-1,0]xf64>
// CHECK-NEXT:   %zbuf_t = stencil.load %zbuf : !stencil.field<[-1,63]xf64> -> !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:   %2 = "dmp.swap"(%zbuf_t) {strategy = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, swaps = []} : (!stencil.temp<[-1,63]xf64>) -> !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:   %res = stencil.apply(%inarg = %0 : !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>, %ybuf_arg = %1 : !stencil.temp<[-1,0]xf64>, %zbuf_arg = %2 : !stencil.temp<[-1,63]xf64>) -> (!stencil.temp<[0,1]x[0,1]x[0,64]xf64>) {
// CHECK-NEXT:     %3 = stencil.access %ybuf_arg[_, -1, _] : !stencil.temp<[-1,0]xf64>
// CHECK-NEXT:     %4 = stencil.access %zbuf_arg[_, _, -1] : !stencil.temp<[-1,63]xf64>
// CHECK-NEXT:     %5 = stencil.access %inarg[0, -1, -1] : !stencil.temp<[0,1]x[-1,0]x[-1,63]xf64>
// CHECK-NEXT:     %6 = arith.addf %3, %4 : f64
// CHECK-NEXT:     %7 = arith.addf %5, %6 : f64
// CHECK-NEXT:     stencil.return %7 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   stencil.store %res to %out(<[0, 0, 0], [1, 1, 64]>) : !stencil.temp<[0,1]x[0,1]x[0,64]xf64> to !stencil.field<[0,1]x[-1,1]x[-1,64]xf64>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }


// RESTRICT:   Cannot restrict stencil programs with different dimensionality
