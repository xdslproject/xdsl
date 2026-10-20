// RUN: xdsl-opt %s | $XDSL_MLIR_OPT --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 10 : i32} : () -> i32
  %4 = "arith.constant"() {"value" = 100 : i32} : () -> i32
  %7 = "scf.parallel"(%0, %1, %2, %3) ({
    ^bb0(%8: index):
      scf.reduce(%4 : i32) {
      ^bb1(%9: i32, %10: i32):
        %11 = "arith.addi"(%9, %10) : (i32, i32) -> i32
        scf.reduce.return %11 : i32
      }
    }) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (index, index, index, i32) -> i32

  %initial_f32 = arith.constant 1.5 : f32
  %sum, %product = "scf.parallel"(%0, %1, %2, %3, %initial_f32) ({
  ^bb0(%j: index):
    scf.reduce(%4, %initial_f32 : i32, f32) {
    ^bb0(%ilhs: i32, %irhs: i32):
      %iadd = arith.addi %ilhs, %irhs : i32
      scf.reduce.return %iadd : i32
    }, {
    ^bb0(%flhs: f32, %frhs: f32):
      %fmul = arith.mulf %flhs, %frhs : f32
      scf.reduce.return %fmul : f32
    } {tag = "sum_product"}
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 2>} : (index, index, index, i32, f32) -> (i32, f32)
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT:  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:  %1 = "arith.constant"() <{value = 1000 : index}> : () -> index
// CHECK-NEXT:  %2 = "arith.constant"() <{value = 3 : index}> : () -> index
// CHECK-NEXT:  %3 = "arith.constant"() <{value = 10 : i32}> : () -> i32
// CHECK-NEXT:  %4 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:  %5 = "scf.parallel"(%0, %1, %2, %3) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
// CHECK-NEXT:  ^bb0(%arg5: index):
// CHECK-NEXT:    "scf.reduce"(%4) ({
// CHECK-NEXT:    ^bb0(%arg6: i32, %arg7: i32):
// CHECK-NEXT:      %6 = "arith.addi"(%arg6, %arg7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-NEXT:      "scf.reduce.return"(%6) : (i32) -> ()
// CHECK-NEXT:    }) : (i32) -> ()
// CHECK-NEXT:  }) : (index, index, index, i32) -> i32
// CHECK-NEXT:  %7 = "arith.constant"() <{value = 1.500000e+00 : f32}> : () -> f32
// CHECK-NEXT:  %8, %9 = "scf.parallel"(%0, %1, %2, %3, %7) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK-NEXT:  ^bb0(%arg0: index):
// CHECK-NEXT:    "scf.reduce"(%4, %7) ({
// CHECK-NEXT:    ^bb0(%arg3: i32, %arg4: i32):
// CHECK-NEXT:      %10 = "arith.addi"(%arg3, %arg4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-NEXT:      "scf.reduce.return"(%10) : (i32) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:    ^bb0(%arg1: f32, %arg2: f32):
// CHECK-NEXT:      %11 = "arith.mulf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:      "scf.reduce.return"(%11) : (f32) -> ()
// CHECK-NEXT:    }) {tag = "sum_product"} : (i32, f32) -> ()
// CHECK-NEXT:  }) : (index, index, index, i32, f32) -> (i32, f32)
// CHECK-NEXT:}) : () -> ()
