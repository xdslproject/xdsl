// RUN: xdsl-opt -p convert-memref-to-riscv --split-input-file --verify-diagnostics %s | filecheck %s

// CHECK: memref.global value size not divisible by xlen (32) not yet supported.
"memref.global"() {"sym_name" = "global_array", "type" = memref<3xi8>, "sym_visibility" = "public", "initial_value" = dense<[1, 2, 3]> : tensor<3xi8>} : () -> ()

// -----

// CHECK: Unsupported memref.global initial value: unit
"memref.global"() {"sym_name" = "uninit_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value"  } : () -> ()
