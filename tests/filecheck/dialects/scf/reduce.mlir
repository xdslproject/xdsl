// RUN: xdsl-opt %s --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 10.2 : f32} : () -> f32
  %4 = "arith.constant"() {"value" = 18.1 : f32} : () -> f32
  %5 = "scf.parallel"(%0, %1, %2, %3) ({
    ^0(%8 : index):
      "scf.reduce"(%4) ({
      ^1(%9 : f32, %10 : f32):
        %11 = "arith.addf"(%9, %10) : (f32, f32) -> f32
        "scf.reduce.return"(%11) : (f32) -> ()
      }) : (f32) -> ()
      "scf.yield"() : () -> ()
    }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:   %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
// CHECK-NEXT:   %2 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:   %3 = "arith.constant"() {"value" = 10.2 : f32} : () -> f32
// CHECK-NEXT:   %4 = "arith.constant"() {"value" = 18.1 : f32} : () -> f32
// CHECK-NEXT:   %5 = "scf.parallel"(%0, %1, %2, %3) ({
// CHECK-NEXT:   ^0(%6 : index):
// CHECK-NEXT:     "scf.reduce"(%4) ({
// CHECK-NEXT:     ^1(%7 : f32, %8 : f32):
// CHECK-NEXT:       %9 = "arith.addf"(%7, %8) : (f32, f32) -> f32
// CHECK-NEXT:       "scf.reduce.return"(%9) : (f32) -> ()
// CHECK-NEXT:     }) : (f32) -> ()
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
// CHECK-NEXT: }) : () -> ()
