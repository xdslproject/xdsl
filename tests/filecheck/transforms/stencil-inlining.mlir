// RUN: xdsl-opt %s --split-input-file -p stencil-inlining | filecheck %s

func.func @simple(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
  %2 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
    %5 = stencil.access %arg2 [-1, 0, 0] : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
    %6 = stencil.access %arg2 [1, 0, 0] : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
    %7 = arith.addf %5, %6 : f64
    %8 = stencil.store_result %7 : !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  }
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>, %arg3 = %3 : !stencil.temp<[1,65]x[2,66]x[3,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %5 = stencil.access %arg2 [0, 0, 0] : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
    %6 = stencil.access %arg3 [1, 2, 3] : !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    %7 = arith.addf %5, %6 : f64
    %8 = stencil.store_result %7 : !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  }
  stencil.store %4 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @simple(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:      %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %2 = stencil.access %arg2[0, 0, 0] : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %3 = stencil.access %arg2[0, 2, 3] : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %4 = stencil.access %arg2[2, 2, 3] : !stencil.temp<[0,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:        %6 = arith.addf %2, %5 : f64
// CHECK-NEXT:        %7 = stencil.store_result %6 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %7 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %1 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @simple_index(%arg0: f64, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
  %1 = stencil.apply (%arg2 = %arg0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
    %3 = stencil.index 2 <[2, -1, 1]>
    %c20 = arith.constant 20 : index
    %cst = arith.constant 0.000000e+00 : f64
    %4 = arith.cmpi slt, %3, %c20 : index
    %5 = arith.select %4, %arg2, %cst : f64
    %6 = stencil.store_result %5 : !stencil.result<f64>
    stencil.return %6 : !stencil.result<f64>
  }
  %2 = stencil.apply (%arg2 = %arg0 : f64, %arg3 = %1 : !stencil.temp<[1,65]x[2,66]x[3,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %7 = stencil.access %arg3 [1, 2, 3] : !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    %8 = arith.addf %7, %arg2 : f64
    %9 = stencil.store_result %8 : !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  }
  stencil.store %2 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:    func.func @simple_index(%arg0 : f64, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:      %0 = stencil.apply(%arg2 = %arg0 : f64) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:      %1 = stencil.index 2 <[3, 1, 4]>
// CHECK-NEXT:      %c20 = arith.constant 20 : index
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      %2 = arith.cmpi slt, %1, %c20 : index
// CHECK-NEXT:      %3 = arith.select %2, %arg2, %cst : f64
// CHECK-NEXT:      %4 = arith.addf %3, %arg2 : f64
// CHECK-NEXT:      %5 = stencil.store_result %4 : !stencil.result<f64>
// CHECK-NEXT:      stencil.return %5 : !stencil.result<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    stencil.store %0 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

// -----

func.func @simple_ifelse(%arg0: f64, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
    %1 = stencil.apply (%arg2 = %arg0 : f64) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %true = arith.constant true
    %3 = "scf.if"(%true) ({
      %4 = stencil.store_result %arg2 : !stencil.result<f64>
      scf.yield %4 : !stencil.result<f64>
    }, {
      %4 = stencil.store_result %arg2 : !stencil.result<f64>
      scf.yield %4 : !stencil.result<f64>
    }) : (i1) -> (!stencil.result<f64>)
    stencil.return %3 : !stencil.result<f64>
  }
  %2 = stencil.apply (%arg2 = %1 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %3 = stencil.access %arg2 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %4 = stencil.store_result %3 : !stencil.result<f64>
    stencil.return %4 : !stencil.result<f64>
  }
  stencil.store %2 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @simple_ifelse(%arg0 : f64, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:       %0 = stencil.apply(%arg2 = %arg0 : f64) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:       %true = arith.constant true
// CHECK-NEXT:       %1 = scf.if %true -> (f64) {
// CHECK-NEXT:         scf.yield %arg2 : f64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %arg2 : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       %2 = stencil.store_result %1 : !stencil.result<f64>
// CHECK-NEXT:       stencil.return %2 : !stencil.result<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     stencil.store %0 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }

// -----

func.func @multiple_edges(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg2: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
  %3 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
  %4:2 = stencil.apply (%arg3 = %3 : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %7 = stencil.access %arg3 [-1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %8 = stencil.access %arg3 [1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %9 = stencil.store_result %7 : !stencil.result<f64>
    %10 = stencil.store_result %8 : !stencil.result<f64>
    stencil.return %9, %10 : !stencil.result<f64>, !stencil.result<f64>
  }
  %5 = stencil.load %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  %6 = stencil.apply (%arg3 = %3 : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>, %arg4 = %4#0 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>, %arg5 = %4#1 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>, %arg6 = %5 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %7 = stencil.access %arg3 [0, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %8 = stencil.access %arg4 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %9 = stencil.access %arg5 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %10 = stencil.access %arg6 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %11 = arith.addf %7, %8 : f64
    %12 = arith.addf %9, %10 : f64
    %13 = arith.addf %11, %12 : f64
    %14 = stencil.store_result %13 : !stencil.result<f64>
    stencil.return %14 : !stencil.result<f64>
  }
  stencil.store %6 to %arg2(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @multiple_edges(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg2 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.load %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %2 = stencil.apply(%arg3 = %0 : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>, %arg6 = %1 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %arg3[0, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %4 = stencil.access %arg3[-1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %5 = stencil.access %arg3[1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %6 = stencil.access %arg6[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %7 = arith.addf %3, %4 : f64
// CHECK-NEXT:        %8 = arith.addf %5, %6 : f64
// CHECK-NEXT:        %9 = arith.addf %7, %8 : f64
// CHECK-NEXT:        %10 = stencil.store_result %9 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %10 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %2 to %arg2(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @avoid_redundant(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
  %2 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %5 = stencil.access %arg2 [-1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %6 = stencil.access %arg2 [1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %7 = arith.addf %5, %6 : f64
    %8 = stencil.store_result %7 : !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  }
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %5 = stencil.access %arg2 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %6 = stencil.access %arg2 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %7 = arith.addf %5, %6 : f64
    %8 = stencil.store_result %7 : !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  }
  stencil.store %4 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @avoid_redundant(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %2 = stencil.access %arg2[-1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %3 = stencil.access %arg2[1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %4 = arith.addf %2, %3 : f64
// CHECK-NEXT:        %5 = stencil.access %arg2[-1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %6 = stencil.access %arg2[1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %7 = arith.addf %5, %6 : f64
// CHECK-NEXT:        %8 = arith.addf %4, %7 : f64
// CHECK-NEXT:        %9 = stencil.store_result %8 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %9 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %1 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @reroute(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg2: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
  %3 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,65]x[0,66]x[0,63]xf64>) {
    %6 = stencil.access %arg3 [-1, 0, 0] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
    %7 = stencil.access %arg3 [1, 0, 0] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
    %8 = arith.addf %6, %7 : f64
    %9 = stencil.store_result %8 : !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  }
  %5 = stencil.apply (%arg4 = %4 : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %6 = stencil.access %arg4 [0, 0, 0] : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
    %7 = stencil.access %arg4 [1, 2, 3] : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
    %8 = arith.addf %6, %7 : f64
    %9 = stencil.store_result %8 : !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  }
  stencil.store %4 to %arg1(<[0, 0, 0], [65, 66, 63]>) : !stencil.temp<[0,65]x[0,66]x[0,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  stencil.store %5 to %arg2(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @reroute(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg2 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:      %1, %2 = stencil.apply(%arg3 = %0 : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,65]x[0,66]x[0,63]xf64>, !stencil.temp<[0,65]x[0,66]x[0,63]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %arg3[-1, 0, 0] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %4 = stencil.access %arg3[1, 0, 0] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:        %6 = stencil.access %arg3[0, 2, 3] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %7 = stencil.access %arg3[2, 2, 3] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %8 = arith.addf %6, %7 : f64
// CHECK-NEXT:        %9 = arith.addf %5, %8 : f64
// CHECK-NEXT:        %10 = stencil.store_result %9 : !stencil.result<f64>
// CHECK-NEXT:        %11 = stencil.access %arg3[-1, 0, 0] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %12 = stencil.access %arg3[1, 0, 0] : !stencil.temp<[-1,66]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %13 = arith.addf %11, %12 : f64
// CHECK-NEXT:        stencil.return %10, %13 : !stencil.result<f64>, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %2 to %arg1(<[0, 0, 0], [65, 66, 63]>) : !stencil.temp<[0,65]x[0,66]x[0,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      stencil.store %1 to %arg2(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,65]x[0,66]x[0,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @root(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg2: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
  %3 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %6 = stencil.access %arg3 [0, 0, 0] : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
    %7 = stencil.store_result %6 : !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  %5 = stencil.apply (%arg3 = %3 : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %6 = stencil.access %arg3 [1, 2, 3] : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
    %7 = stencil.store_result %6 : !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  stencil.store %4 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  stencil.store %5 to %arg2(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @root(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg2 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
// CHECK-NEXT:      %1, %2 = stencil.apply(%arg3 = %0 : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %3 = stencil.access %arg3[1, 2, 3] : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        %4 = stencil.store_result %3 : !stencil.result<f64>
// CHECK-NEXT:        %5 = stencil.access %arg3[0, 0, 0] : !stencil.temp<[0,65]x[0,66]x[0,63]xf64>
// CHECK-NEXT:        stencil.return %4, %5 : !stencil.result<f64>, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %2 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      stencil.store %1 to %arg2(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @dyn_access(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
  %2 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-2,66]x[-1,65]x[-2,60]xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<[-2,66]x[-1,65]x[-2,60]xf64>) -> (!stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>) {
    %6 = stencil.access %arg2 [0, 0, -1] : !stencil.temp<[-2,66]x[-1,65]x[-2,60]xf64>
    %7 = stencil.store_result %6 : !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>) -> (!stencil.temp<[-1,65]x[0,64]x[0,60]xf64>) {
    %7 = stencil.index 0 <[0, 0, 0]>
    %8 = stencil.dyn_access %arg3[%7, %7, %7] in <[-1, -1, -1]> : <[1, 1, 1]> : !stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>
    %9 = stencil.store_result %8 : !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  }
  %5 = stencil.apply (%arg4 = %4 : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %10 = stencil.access %arg4 [-1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %11 = stencil.access %arg4 [1, 0, 0] : !stencil.temp<[-1,65]x[0,64]x[0,60]xf64>
    %12 = arith.addf %11, %10 : f64
    %13 = stencil.store_result %12 : !stencil.result<f64>
    stencil.return %13 : !stencil.result<f64>
  }
  stencil.store %5 to %arg1(<[0, 0, 0], [64, 64, 60]>): !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @dyn_access(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[-2,66]x[-1,65]x[-2,60]xf64>
// CHECK-NEXT:      %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[-2,66]x[-1,65]x[-2,60]xf64>) -> (!stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>) {
// CHECK-NEXT:        %2 = stencil.access %arg2[0, 0, -1] : !stencil.temp<[-2,66]x[-1,65]x[-2,60]xf64>
// CHECK-NEXT:        %3 = stencil.store_result %2 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %3 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %2 = stencil.apply(%arg3 = %1 : !stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %3 = stencil.index 0 <[-1, 0, 0]>
// CHECK-NEXT:        %4 = stencil.dyn_access %arg3[%3, %3, %3] in <[-2, -1, -1]> : <[0, 1, 1]> : !stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>
// CHECK-NEXT:        %5 = stencil.index 0 <[1, 0, 0]>
// CHECK-NEXT:        %6 = stencil.dyn_access %arg3[%5, %5, %5] in <[0, -1, -1]> : <[2, 1, 1]> : !stencil.temp<[-2,66]x[-1,65]x[-1,61]xf64>
// CHECK-NEXT:        %7 = arith.addf %6, %4 : f64
// CHECK-NEXT:        %8 = stencil.store_result %7 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %8 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %2 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @simple_buffer(%arg0: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) attributes {stencil.program} {
  %2 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %6 = stencil.access %arg2 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %7 = stencil.store_result %6 : !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  %4 = stencil.buffer %3 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  %5 = stencil.apply (%arg2 = %4 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %6 = stencil.access %arg2 [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %7 = stencil.store_result %6 : !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  stencil.store %5 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
  return
}

// CHECK:         func.func @simple_buffer(%arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>, %arg1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>)  attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.load %arg0 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %2 = stencil.access %arg2[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %3 = stencil.store_result %2 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %3 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %2 = stencil.buffer %1 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%arg2 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %4 = stencil.access %arg2[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %5 = stencil.store_result %4 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %5 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %arg1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
