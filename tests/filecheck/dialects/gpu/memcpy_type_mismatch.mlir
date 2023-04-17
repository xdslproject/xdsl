// RUN: xdsl-opt --verify-diagnostic %s | filecheck %s

"builtin.module"() ({
    %memref = "memref.alloc"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<10x10xi32>
    %ten = "arith.constant"() {"value" = 10 : index} : () -> index
    %gmemref = "gpu.alloc"(%ten, %ten) {"operand_segment_sizes" = array<i32: 0, 2, 0>} : (index, index) -> memref<?x?xi32>

    "gpu.memcpy"(%memref, %gmemref) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<10x10xi32>, memref<?x?xi32>) -> ()

}) : () -> ()

// CHECK: Expected memref<10x10xi32>, got memref<?x?xi32>. gpu.memcpy source and destination types must match.