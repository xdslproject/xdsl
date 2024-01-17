// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  "memref.global"() {"alignment" = 64 : i32, "sym_name" = "wrong_alignment_type", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
}

// CHECK: Expected attribute i64 but got i32

// -----

builtin.module {
  "memref.global"() {"alignment" = 65 : i64, "sym_name" = "non_power_of_two_alignment", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
}

// CHECK: Alignment attribute 65 is not a power of 2
