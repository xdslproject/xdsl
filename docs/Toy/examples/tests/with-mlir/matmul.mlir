"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x2xf64>
    %1 = "arith.constant"() {"value" = 1.000000e+00 : f64} : () -> f64
    %2 = "arith.constant"() {"value" = 2.000000e+00 : f64} : () -> f64
    %3 = "arith.constant"() {"value" = 3.000000e+00 : f64} : () -> f64
    %4 = "arith.constant"() {"value" = 4.000000e+00 : f64} : () -> f64
    %5 = "arith.constant"() {"value" = 5.000000e+00 : f64} : () -> f64
    %6 = "arith.constant"() {"value" = 6.000000e+00 : f64} : () -> f64
    %7 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
    %8 = "arith.constant"() {"value" = 1.000000e+00 : f64} : () -> f64
    %9 = "arith.constant"() {"value" = 2.000000e+00 : f64} : () -> f64
    %10 = "arith.constant"() {"value" = 3.000000e+00 : f64} : () -> f64
    %11 = "arith.constant"() {"value" = 4.000000e+00 : f64} : () -> f64
    %12 = "arith.constant"() {"value" = 5.000000e+00 : f64} : () -> f64
    %13 = "arith.constant"() {"value" = 6.000000e+00 : f64} : () -> f64
    %14 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xf64>
    "affine.store"(%8, %14) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%9, %14) {"map" = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%10, %14) {"map" = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%11, %14) {"map" = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%12, %14) {"map" = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%13, %14) {"map" = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%1, %7) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<3x2xf64>) -> ()
    "affine.store"(%2, %7) {"map" = affine_map<() -> (0, 1)>} : (f64, memref<3x2xf64>) -> ()
    "affine.store"(%3, %7) {"map" = affine_map<() -> (1, 0)>} : (f64, memref<3x2xf64>) -> ()
    "affine.store"(%4, %7) {"map" = affine_map<() -> (1, 1)>} : (f64, memref<3x2xf64>) -> ()
    "affine.store"(%5, %7) {"map" = affine_map<() -> (2, 0)>} : (f64, memref<3x2xf64>) -> ()
    "affine.store"(%6, %7) {"map" = affine_map<() -> (2, 1)>} : (f64, memref<3x2xf64>) -> ()
    "affine.for"() ({
    ^0(%15 : index):
      "affine.for"() ({
      ^1(%16 : index):
        %10 = arith.constant 0.0 : f64
        %11 = "affine.for"() ({
        ^2(%k : index):
            %17 = "affine.load"(%14, %15, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64
            %18 = "affine.load"(%14, %15, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64
            %19 = "arith.addf"(%17, %18) : (f64, f64) -> f64
            "affine.yield"() : () -> ()
        }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (3)>, "step" = 1 : index} : () -> ()
        "affine.store"(%19, %0, %15, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<2x3xf64>, index, index) -> ()
      }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (2)>, "step" = 1 : index} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (2)>, "step" = 1 : index} : () -> ()
    "printf.print_format"(%0) {"format_str" = "{}"} : (memref<2x3xf64>) -> ()
    "printf.print_format"(%7) {"format_str" = "{}"} : (memref<3x2xf64>) -> ()
    "memref.dealloc"(%14) : (memref<2x3xf64>) -> ()
    "memref.dealloc"(%7) : (memref<3x2xf64>) -> ()
    "memref.dealloc"(%0) : (memref<2x3xf64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
}) : () -> ()
