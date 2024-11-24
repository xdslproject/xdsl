// RUN: xdsl-run --wgpu --index-bitwidth=32 %s | filecheck %s

builtin.module attributes {gpu.container_module} {
  "gpu.module"() ({
    "gpu.func"() ({
    ^0(%arg : memref<4x4xindex>):
      %0 = "arith.constant"() {"value" = 2 : index} : () -> index
      %1 = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
      %2 = "gpu.global_id"() {"dimension" = #gpu<dim y>} : () -> index
      %3 = "arith.constant"() {"value" = 4 : index} : () -> index
      %4 = "arith.muli"(%1, %3) : (index, index) -> index
      %5 = "arith.addi"(%4, %2) : (index, index) -> index
      "memref.store"(%5, %arg, %1, %2) {"nontemporal" = false} : (index, memref<4x4xindex>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (memref<4x4xindex>) -> (),
        "gpu.kernel",
        "sym_name" = "fill"
       } : () -> ()
    "gpu.func"() ({
    ^0(%arg : memref<4x4xindex>):
      %0 = "arith.constant"() {"value" = 1 : index} : () -> index
      %1 = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
      %2 = "gpu.global_id"() {"dimension" = #gpu<dim y>} : () -> index
      %3 = "memref.load"(%arg, %1, %2) {"nontemporal" = false} : (memref<4x4xindex>, index, index) -> (index)
      %4 = "arith.addi"(%3, %0) : (index, index) -> index
      "memref.store"(%4, %arg, %1, %2) {"nontemporal" = false} : (index, memref<4x4xindex>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (memref<4x4xindex>) -> (),
        "gpu.kernel",
        "sym_name" = "inc"
       } : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "gpu"} : () -> ()
  func.func @main() -> index {
    %four = "arith.constant"() {"value" = 4 : index} : () -> index
    %one = "arith.constant"() {"value" = 1 : index} : () -> index
    %memref = "gpu.alloc"() {"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0, 0>} : () -> memref<4x4xindex>
    "gpu.launch_func"(%four, %four, %one, %one, %one, %one, %memref) {"operandSegmentSizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0>, "kernel" = @gpu::@fill} : (index, index, index, index, index, index, memref<4x4xindex>) -> ()
    "gpu.launch_func"(%four, %four, %one, %one, %one, %one, %memref) {"operandSegmentSizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0>, "kernel" = @gpu::@inc} : (index, index, index, index, index, index, memref<4x4xindex>) -> ()
    %hmemref = "memref.alloc"() {"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>} : () -> memref<4x4xindex>
    "gpu.memcpy"(%hmemref, %memref) {"operandSegmentSizes" = array<i32: 0, 1, 1>} : (memref<4x4xindex>, memref<4x4xindex>) -> ()
    printf.print_format "Result : {}", %hmemref : memref<4x4xindex>
    %zero  = "arith.constant"() {"value" = 0 : index} : () -> (index)
    "func.return"(%zero) : (index) -> ()
  }
}

// CHECK-NEXT{LITERAL}: Result : [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
