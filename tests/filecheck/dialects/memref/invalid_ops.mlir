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
