// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  "memref.global"() {"alignment" = 64 : i32, "sym_name" = "wrong_alignment_type", "type" = memref<1xi32>, "initial_value" = dense<0> : tensor<1xi32>, "sym_visibility" = "public"} : () -> ()
}

// CHECK: Expected attribute i64 but got i32

// -----

builtin.module {
  "memref.global"() {"alignment" = 65 : i64, "sym_name" = "non_power_of_two_alignment", "type" = memref<1xi32>, "initial_value" = dense<0> : tensor<1xi32>, "sym_visibility" = "public"} : () -> ()
}

// CHECK: Alignment attribute 65 is not a power of 2

// -----

"func.func"() ({
    %0 = "memref.alloc"() {"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}: () -> memref<10x2xindex>
    %1 = "memref.collapse_shape"(%0) {"reassociation" = [[0 : i32 , 1 : i32]]} : (memref<10x2xindex>) -> memref<20xindex>
    "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "invalid_reassociation"} : () -> ()



// CHECK: Expected attribute i64

// -----

"func.func"() ({
    %0 = "memref.alloc"() {"alignment" = 64 : i64, "operandSegmentSizes" = array<i32: 0, 0>}: () -> memref<20xindex>
    %1 = "memref.expand_shape"(%0) {"reassociation" = [[0 : i32 , 1 : i32]]} : (memref<20xindex>) -> memref<2x10xindex>
    "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "invalid_reassociation"} : () -> ()

// CHECK: Expected attribute i64
