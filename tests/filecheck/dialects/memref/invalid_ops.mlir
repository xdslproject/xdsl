// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  "memref.global"() {"alignment" = 64 : i32, "sym_name" = "wrong_alignment_type", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
}

// CHECK: Invalid value 32, expected 64

// -----

builtin.module {
  "memref.global"() {"alignment" = 65 : i64, "sym_name" = "non_power_of_two_alignment", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
}

// CHECK: Alignment attribute 65 is not a power of 2

// -----

"func.func"() ({
    %0 = "memref.alloc"() {"alignment" = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}: () -> memref<10x2xindex>
    %1 = "memref.collapse_shape"(%0) {"reassociation" = [[0 : i32 , 1 : i32]]} : (memref<10x2xindex>) -> memref<20xindex>
    "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "invalid_reassociation"} : () -> ()



// CHECK: Invalid value 32, expected 64

// -----

"func.func"() ({
    %0 = "memref.alloc"() {"alignment" = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}: () -> memref<20xindex>
    %1 = "memref.expand_shape"(%0) {"reassociation" = [[0 : i32 , 1 : i32]]} : (memref<20xindex>) -> memref<2x10xindex>
    "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "invalid_reassociation"} : () -> ()

// CHECK: Invalid value 32, expected 64


// -----

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.reinterpret_cast"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 5>, static_strides = array<i64: 1>}> : (memref<10x2xindex>) -> memref<10x2xindex, strided<[1, 1]>>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: Expected 2 size values but got 1


// -----

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.reinterpret_cast"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<10x2xindex, strided<[1, 1]>>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: Expected result type with size = 5 instead of 10 in dim = 0

// -----

// Mismatched sizes in size arguments

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.reinterpret_cast"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<5x4xindex, strided<[1, 1]>>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: The number of dynamic positions passed as values (0) does not match the number of dynamic position markers (2) in the size arguments.

// -----

// Mismatched sizes in offset arguments

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.reinterpret_cast"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<5x4xindex, strided<[1, 1]>>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: The number of dynamic positions passed as values (0) does not match the number of dynamic position markers (2) in the offset arguments.

// -----

// Mismatched sizes in stride arguments

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.reinterpret_cast"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<10x2xindex>) -> memref<5x4xindex, strided<[1, 1]>>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: The number of dynamic positions passed as values (0) does not match the number of dynamic position markers (2) in the stride arguments.

// -----

// Mismatched sizes in size arguments

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.subview"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<5x4xindex>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: The number of dynamic positions passed as values (0) does not match the number of dynamic position markers (2) in the size arguments.

// -----

// Mismatched sizes in offset arguments

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.subview"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<5x4xindex>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: The number of dynamic positions passed as values (0) does not match the number of dynamic position markers (2) in the offset arguments.

// -----

// Mismatched sizes in stride arguments

"func.func"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
  %1 = "memref.subview"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<10x2xindex>) -> memref<5x4xindex>
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "mismatched_sizes"} : () -> ()

// CHECK: The number of dynamic positions passed as values (0) does not match the number of dynamic position markers (2) in the stride arguments.

// -----

// memref.view source element type must be i8

"builtin.module"() ({
  %src = "test.op"() : () -> memref<2048xi32>
  %off = "test.op"() : () -> index
  %v = "memref.view"(%src, %off) : (memref<2048xi32>, index) -> memref<64x4xf32>
}) : () -> ()

// CHECK: Expected attribute i8 but got i32

// -----

// memref.view source must be 1-D

"builtin.module"() ({
  %src = "test.op"() : () -> memref<64x32xi8>
  %off = "test.op"() : () -> index
  %v = "memref.view"(%src, %off) : (memref<64x32xi8>, index) -> memref<64x4xf32>
}) : () -> ()

// CHECK: memref.view source must be a 1-D memref of i8

// -----

// memref.view source must have identity layout

"builtin.module"() ({
  %src = "test.op"() : () -> memref<2048xi8, strided<[1], offset: 0>>
  %off = "test.op"() : () -> index
  %v = "memref.view"(%src, %off) : (memref<2048xi8, strided<[1], offset: 0>>, index) -> memref<64x4xf32>
}) : () -> ()

// CHECK: memref.view source must have identity layout (no layout map)

// -----

// memref.view memory spaces must match

"builtin.module"() ({
  %src = "test.op"() : () -> memref<2048xi8>
  %off = "test.op"() : () -> index
  %v = "memref.view"(%src, %off) : (memref<2048xi8>, index) -> memref<64x4xf32, 1 : i32>
}) : () -> ()

// CHECK: different memory spaces specified for base memref type

// -----

// memref.view dynamic size count must match

"builtin.module"() ({
  %src = "test.op"() : () -> memref<2048xi8>
  %off = "test.op"() : () -> index
  %d0 = "test.op"() : () -> index
  %v = "memref.view"(%src, %off, %d0) : (memref<2048xi8>, index, index) -> memref<64x4xf32>
}) : () -> ()

// CHECK: number of size operands must match number of dynamic dims
