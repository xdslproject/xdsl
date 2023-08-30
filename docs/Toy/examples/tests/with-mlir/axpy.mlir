// RUN: python -m toy %s --emit=scf | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<6xf32>
    %1 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
    %2 = "arith.constant"() {"value" = 2.0 : f32} : () -> f32
    %3 = "arith.constant"() {"value" = 3.0 : f32} : () -> f32
    %4 = "arith.constant"() {"value" = 4.0 : f32} : () -> f32
    %5 = "arith.constant"() {"value" = 5.0 : f32} : () -> f32
    %6 = "arith.constant"() {"value" = 6.0 : f32} : () -> f32
    %7 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<6xf32>
    %8 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
    %9 = "arith.constant"() {"value" = 2.0 : f32} : () -> f32
    %10 = "arith.constant"() {"value" = 3.0 : f32} : () -> f32
    %11 = "arith.constant"() {"value" = 4.0 : f32} : () -> f32
    %12 = "arith.constant"() {"value" = 5.0 : f32} : () -> f32
    %13 = "arith.constant"() {"value" = 6.0 : f32} : () -> f32
    %14 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<6xf32>
    "affine.store"(%8, %14) {"map" = affine_map<() -> (0)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%9, %14) {"map" = affine_map<() -> (1)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%10, %14) {"map" = affine_map<() -> (2)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%11, %14) {"map" = affine_map<() -> (3)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%12, %14) {"map" = affine_map<() -> (4)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%13, %14) {"map" = affine_map<() -> (5)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%1, %7) {"map" = affine_map<() -> (0)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%2, %7) {"map" = affine_map<() -> (1)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%3, %7) {"map" = affine_map<() -> (2)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%4, %7) {"map" = affine_map<() -> (3)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%5, %7) {"map" = affine_map<() -> (4)>} : (f32, memref<6xf32>) -> ()
    "affine.store"(%6, %7) {"map" = affine_map<() -> (5)>} : (f32, memref<6xf32>) -> ()
    "affine.for"() ({
    ^0(%i : index):
        %17 = "affine.load"(%14, %i) {"map" = affine_map<(d0) -> (d0)>} : (memref<6xf32>, index) -> f32
        %18 = "affine.load"(%7, %i) {"map" = affine_map<(d0) -> (d0)>} : (memref<6xf32>, index) -> f32
        %x = "arith.addf"(%17, %18) : (f32, f32) -> f32
        "affine.store"(%x, %0, %i) {"map" = affine_map<(d0) -> (d0)>} : (f32, memref<6xf32>, index) -> ()
        "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (6)>, "step" = 1 : index} : () -> ()
    "printf.print_format"(%0) {"format_str" = "{}"} : (memref<6xf32>) -> ()
    "memref.dealloc"(%14) : (memref<6xf32>) -> ()
    "memref.dealloc"(%7) : (memref<6xf32>) -> ()
    "memref.dealloc"(%0) : (memref<6xf32>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
}) : () -> ()

// CHECK:       [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
