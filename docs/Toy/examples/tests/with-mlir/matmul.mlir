// RUN: python -m toy %s --emit=scf | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x2xf32>
    %1 = "arith.constant"() {"value" = 1 : f32} : () -> f32
    %2 = "arith.constant"() {"value" = 2 : f32} : () -> f32
    %3 = "arith.constant"() {"value" = 3 : f32} : () -> f32
    %4 = "arith.constant"() {"value" = 4 : f32} : () -> f32
    %5 = "arith.constant"() {"value" = 5 : f32} : () -> f32
    %6 = "arith.constant"() {"value" = 6 : f32} : () -> f32
    %7 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf32>
    %8 = "arith.constant"() {"value" = 1 : f32} : () -> f32
    %9 = "arith.constant"() {"value" = 2 : f32} : () -> f32
    %10 = "arith.constant"() {"value" = 3 : f32} : () -> f32
    %11 = "arith.constant"() {"value" = 4 : f32} : () -> f32
    %12 = "arith.constant"() {"value" = 5 : f32} : () -> f32
    %13 = "arith.constant"() {"value" = 6 : f32} : () -> f32
    %14 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xf32>
    "affine.store"(%8, %14) {"map" = affine_map<() -> (0, 0)>} : (f32, memref<2x3xf32>) -> ()
    "affine.store"(%9, %14) {"map" = affine_map<() -> (0, 1)>} : (f32, memref<2x3xf32>) -> ()
    "affine.store"(%10, %14) {"map" = affine_map<() -> (0, 2)>} : (f32, memref<2x3xf32>) -> ()
    "affine.store"(%11, %14) {"map" = affine_map<() -> (1, 0)>} : (f32, memref<2x3xf32>) -> ()
    "affine.store"(%12, %14) {"map" = affine_map<() -> (1, 1)>} : (f32, memref<2x3xf32>) -> ()
    "affine.store"(%13, %14) {"map" = affine_map<() -> (1, 2)>} : (f32, memref<2x3xf32>) -> ()
    "affine.store"(%1, %7) {"map" = affine_map<() -> (0, 0)>} : (f32, memref<3x2xf32>) -> ()
    "affine.store"(%2, %7) {"map" = affine_map<() -> (0, 1)>} : (f32, memref<3x2xf32>) -> ()
    "affine.store"(%3, %7) {"map" = affine_map<() -> (1, 0)>} : (f32, memref<3x2xf32>) -> ()
    "affine.store"(%4, %7) {"map" = affine_map<() -> (1, 1)>} : (f32, memref<3x2xf32>) -> ()
    "affine.store"(%5, %7) {"map" = affine_map<() -> (2, 0)>} : (f32, memref<3x2xf32>) -> ()
    "affine.store"(%6, %7) {"map" = affine_map<() -> (2, 1)>} : (f32, memref<3x2xf32>) -> ()
    "affine.for"() ({
    ^0(%15 : index):
      "affine.for"() ({
      ^1(%16 : index):
        %init = arith.constant 0 : f32
        %cell = "affine.for"(%init) ({
        ^2(%k : index, %acc : f32):
            %17 = "affine.load"(%14, %15, %k) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf32>, index, index) -> f32
            %18 = "affine.load"(%7, %k, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<3x2xf32>, index, index) -> f32
            %x = "arith.mulf"(%17, %18) : (f32, f32) -> f32
            %acc_new = "arith.addf"(%acc, %x) : (f32, f32) -> f32
            "affine.yield"(%acc_new) : (f32) -> ()
        }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (3)>, "step" = 1 : index} : (f32) -> (f32)
        "affine.store"(%cell, %0, %15, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (f32, memref<2x2xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (2)>, "step" = 1 : index} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (2)>, "step" = 1 : index} : () -> ()
    "printf.print_format"(%0) {"format_str" = "{}"} : (memref<2x2xf32>) -> ()
    "memref.dealloc"(%14) : (memref<2x3xf32>) -> ()
    "memref.dealloc"(%7) : (memref<3x2xf32>) -> ()
    "memref.dealloc"(%0) : (memref<2x2xf32>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
}) : () -> ()

// CHECK:       [[22.0, 28.0], [49.0, 64.0]]
