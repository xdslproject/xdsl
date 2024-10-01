// RUN: xdsl-opt -p stencil-shape-minimize --split-input-file %s | filecheck %s

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
