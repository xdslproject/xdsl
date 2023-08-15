// RUN: python -m toy %s --emit=scf | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x2xi32>
    %1 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    %2 = "arith.constant"() {"value" = 2 : i32} : () -> i32
    %3 = "arith.constant"() {"value" = 3 : i32} : () -> i32
    %4 = "arith.constant"() {"value" = 4 : i32} : () -> i32
    %5 = "arith.constant"() {"value" = 5 : i32} : () -> i32
    %6 = "arith.constant"() {"value" = 6 : i32} : () -> i32
    %7 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xi32>
    %8 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    %9 = "arith.constant"() {"value" = 2 : i32} : () -> i32
    %10 = "arith.constant"() {"value" = 3 : i32} : () -> i32
    %11 = "arith.constant"() {"value" = 4 : i32} : () -> i32
    %12 = "arith.constant"() {"value" = 5 : i32} : () -> i32
    %13 = "arith.constant"() {"value" = 6 : i32} : () -> i32
    %14 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xi32>
    "affine.store"(%8, %14) {"map" = affine_map<() -> (0, 0)>} : (i32, memref<2x3xi32>) -> ()
    "affine.store"(%9, %14) {"map" = affine_map<() -> (0, 1)>} : (i32, memref<2x3xi32>) -> ()
    "affine.store"(%10, %14) {"map" = affine_map<() -> (0, 2)>} : (i32, memref<2x3xi32>) -> ()
    "affine.store"(%11, %14) {"map" = affine_map<() -> (1, 0)>} : (i32, memref<2x3xi32>) -> ()
    "affine.store"(%12, %14) {"map" = affine_map<() -> (1, 1)>} : (i32, memref<2x3xi32>) -> ()
    "affine.store"(%13, %14) {"map" = affine_map<() -> (1, 2)>} : (i32, memref<2x3xi32>) -> ()
    "affine.store"(%1, %7) {"map" = affine_map<() -> (0, 0)>} : (i32, memref<3x2xi32>) -> ()
    "affine.store"(%2, %7) {"map" = affine_map<() -> (0, 1)>} : (i32, memref<3x2xi32>) -> ()
    "affine.store"(%3, %7) {"map" = affine_map<() -> (1, 0)>} : (i32, memref<3x2xi32>) -> ()
    "affine.store"(%4, %7) {"map" = affine_map<() -> (1, 1)>} : (i32, memref<3x2xi32>) -> ()
    "affine.store"(%5, %7) {"map" = affine_map<() -> (2, 0)>} : (i32, memref<3x2xi32>) -> ()
    "affine.store"(%6, %7) {"map" = affine_map<() -> (2, 1)>} : (i32, memref<3x2xi32>) -> ()
    "affine.for"() ({
    ^0(%15 : index):
      "affine.for"() ({
      ^1(%16 : index):
        %init = arith.constant 0 : i32
        %cell = "affine.for"(%init) ({
        ^2(%k : index, %acc : i32):
            %17 = "affine.load"(%14, %15, %k) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xi32>, index, index) -> i32
            %18 = "affine.load"(%7, %k, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<3x2xi32>, index, index) -> i32
            %x = "arith.muli"(%17, %18) : (i32, i32) -> i32
            %acc_new = "arith.addi"(%acc, %x) : (i32, i32) -> i32
            "affine.yield"(%acc_new) : (i32) -> ()
        }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (3)>, "step" = 1 : index} : (i32) -> (i32)
        "affine.store"(%cell, %0, %15, %16) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (i32, memref<2x2xi32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (2)>, "step" = 1 : index} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (2)>, "step" = 1 : index} : () -> ()
    "printf.print_format"(%0) {"format_str" = "{}"} : (memref<2x2xi32>) -> ()
    "memref.dealloc"(%14) : (memref<2x3xi32>) -> ()
    "memref.dealloc"(%7) : (memref<3x2xi32>) -> ()
    "memref.dealloc"(%0) : (memref<2x2xi32>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
}) : () -> ()

// CHECK:       [[22.0, 28.0], [49.0, 64.0]]
