// RUN: xdsl-opt %s --print-op-generic | filecheck %s


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value"  = 6.000000e+00 : f64} : () -> f64
    %1 = "arith.constant"() {"value"  = 5.000000e+00 : f64} : () -> f64
    %2 = "arith.constant"() {"value"  = 4.000000e+00 : f64} : () -> f64
    %3 = "arith.constant"() {"value"  = 3.000000e+00 : f64} : () -> f64
    %4 = "arith.constant"() {"value"  = 2.000000e+00 : f64} : () -> f64
    %5 = "arith.constant"() {"value"  = 1.000000e+00 : f64} : () -> f64
    %6 = "memref.alloc"() {"operand_segment_sizes"  = array<i32: 0, 0>} : () -> memref<3x2xf64>
    %7 = "memref.alloc"() {"operand_segment_sizes"  = array<i32: 0, 0>} : () -> memref<3x2xf64>
    %8 = "memref.alloc"() {"operand_segment_sizes"  = array<i32: 0, 0>} : () -> memref<2x3xf64>
    "affine.store"(%5, %8) {"map"  = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%4, %8) {"map"  = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%3, %8) {"map"  = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%2, %8) {"map"  = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%1, %8) {"map"  = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%0, %8) {"map"  = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.for"() ({
    ^0(%arg0 : index):
      "affine.for"() ({
      ^1(%arg1 : index):
        %9 = "affine.load"(%8, %arg1, %arg0) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64
        "affine.store"(%9, %7, %arg0, %arg1) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
    "affine.for"() ({
    ^2(%arg0 : index):
      "affine.for"() ({
      ^3(%arg1 : index):
        %9 = "affine.load"(%7, %arg0, %arg1) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (memref<3x2xf64>, index, index) -> f64
        %10 = "arith.mulf"(%9, %9) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
        "affine.store"(%10, %6, %arg0, %arg1) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
    "printf.print_format"(%6) {"format_str" = "{}"} : (memref<3x2xf64>) -> ()
    "memref.dealloc"(%8) : (memref<2x3xf64>) -> ()
    "memref.dealloc"(%7) : (memref<3x2xf64>) -> ()
    "memref.dealloc"(%6) : (memref<3x2xf64>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()


// CHECK:         "builtin.module"() ({
// CHECK-NEXT:      "func.func"() ({
// CHECK-NEXT:        %0 = "arith.constant"() {"value"  = 6.000000e+00 : f64} : () -> f64
// CHECK-NEXT:        %1 = "arith.constant"() {"value"  = 5.000000e+00 : f64} : () -> f64
// CHECK-NEXT:        %2 = "arith.constant"() {"value"  = 4.000000e+00 : f64} : () -> f64
// CHECK-NEXT:        %3 = "arith.constant"() {"value"  = 3.000000e+00 : f64} : () -> f64
// CHECK-NEXT:        %4 = "arith.constant"() {"value"  = 2.000000e+00 : f64} : () -> f64
// CHECK-NEXT:        %5 = "arith.constant"() {"value"  = 1.000000e+00 : f64} : () -> f64
// CHECK-NEXT:        %6 = "memref.alloc"() {"operand_segment_sizes"  = array<i32: 0, 0>} : () -> memref<3x2xf64>
// CHECK-NEXT:        %7 = "memref.alloc"() {"operand_segment_sizes"  = array<i32: 0, 0>} : () -> memref<3x2xf64>
// CHECK-NEXT:        %8 = "memref.alloc"() {"operand_segment_sizes"  = array<i32: 0, 0>} : () -> memref<2x3xf64>
// CHECK-NEXT:        "affine.store"(%5, %8) {"map"  = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%4, %8) {"map"  = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%3, %8) {"map"  = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%2, %8) {"map"  = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%1, %8) {"map"  = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%0, %8) {"map"  = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.for"() ({
// CHECK-NEXT:        ^0(%arg0 : index):
// CHECK-NEXT:          "affine.for"() ({
// CHECK-NEXT:          ^1(%arg1 : index):
// CHECK-NEXT:            %9 = "affine.load"(%8, %arg1, %arg0) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64
// CHECK-NEXT:            "affine.store"(%9, %7, %arg0, %arg1) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
// CHECK-NEXT:            "affine.yield"() : () -> ()
// CHECK-NEXT:          }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
// CHECK-NEXT:          "affine.yield"() : () -> ()
// CHECK-NEXT:        }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
// CHECK-NEXT:        "affine.for"() ({
// CHECK-NEXT:        ^2(%arg0_1 : index):
// CHECK-NEXT:          "affine.for"() ({
// CHECK-NEXT:          ^3(%arg1_1 : index):
// CHECK-NEXT:            %10 = "affine.load"(%7, %arg0_1, %arg1_1) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (memref<3x2xf64>, index, index) -> f64
// CHECK-NEXT:            %11 = "arith.mulf"(%10, %10) {"fastmath" = #arith.fastmath<none>} : (f64, f64) -> f64
// CHECK-NEXT:            "affine.store"(%11, %6, %arg0_1, %arg1_1) {"map"  = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
// CHECK-NEXT:            "affine.yield"() : () -> ()
// CHECK-NEXT:          }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
// CHECK-NEXT:          "affine.yield"() : () -> ()
// CHECK-NEXT:        }) {"lower_bound"   = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
// CHECK-NEXT:        "printf.print_format"(%6) {"format_str" = "{}"} : (memref<3x2xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%8) : (memref<2x3xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%7) : (memref<3x2xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%6) : (memref<3x2xf64>) -> ()
// CHECK-NEXT:        "func.return"() : () -> ()
// CHECK-NEXT:      }) {"function_type" = () -> (), "sym_name" = "main"} : () -> ()
// CHECK-NEXT:    }) : () -> ()
