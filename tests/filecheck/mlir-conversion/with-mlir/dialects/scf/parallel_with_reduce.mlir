// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 10 : i32} : () -> i32
  %4 = "arith.constant"() {"value" = 100 : i32} : () -> i32
  %7 = "scf.parallel"(%0, %1, %2, %3) ({
    ^0(%8 : index):
      "scf.reduce"(%4) ({
      ^1(%9 : i32, %10 : i32):
        %11 = "arith.addi"(%9, %10) : (i32, i32) -> i32
        "scf.reduce.return"(%11) : (i32) -> ()
      }) : (i32) -> ()
      "scf.yield"() : () -> ()
    }) {"operandSegmentSizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, i32) -> i32
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT:  %0 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:  %1 = "arith.constant"() <{"value" = 1000 : index}> : () -> index
// CHECK-NEXT:  %2 = "arith.constant"() <{"value" = 3 : index}> : () -> index
// CHECK-NEXT:  %3 = "arith.constant"() <{"value" = 10 : i32}> : () -> i32
// CHECK-NEXT:  %4 = "arith.constant"() <{"value" = 100 : i32}> : () -> i32
// CHECK-NEXT:  %5 = "scf.parallel"(%0, %1, %2, %3) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1>}> ({
// CHECK-NEXT:  ^0(%arg0 : index):
// CHECK-NEXT:    "scf.reduce"(%4) ({
// CHECK-NEXT:    ^1(%arg1 : i32, %arg2 : i32):
// CHECK-NEXT:      %6 = "arith.addi"(%arg1, %arg2) : (i32, i32) -> i32
// CHECK-NEXT:      "scf.reduce.return"(%6) : (i32) -> ()
// CHECK-NEXT:    }) : (i32) -> ()
// CHECK-NEXT:    "scf.yield"() : () -> ()
// CHECK-NEXT:  }) : (index, index, index, i32) -> i32
// CHECK-NEXT:}) : () -> ()
