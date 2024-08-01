// RUN: XDSL_ROUNDTRIP

builtin.module attributes  {"transform.with_named_sequence", "transform.sequence"} {
  %0 = "test.op"() : () -> !transform.affine_map
  %1 = "test.op"() : () -> !transform.any_op
  %2 = "test.op"() : () -> !transform.any_param
  %3 = "test.op"() : () -> !transform.any_value
  %4 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
  %5 = "test.op"() : () -> !transform.param<i64>
  %6 = "test.op"() : () -> !transform.type
  "transform.named_sequence"() <{"function_type" = (!transform.any_op, !transform.op<"linalg.quantized_matmul">, !transform.op<"linalg.elemwise_binary">) -> (), "sym_name" = "__transform_main"}> ({
  ^0(%arg0 : !transform.any_op, %arg1 : !transform.op<"linalg.quantized_matmul">, %arg2 : !transform.op<"linalg.elemwise_binary">):
    %7 = "transform.cast"(%arg1) : (!transform.op<"linalg.quantized_matmul">) -> !transform.any_op
    %8, %9 = "transform.structured.tile_to_forall_op"(%arg1) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0, 0>, "static_tile_sizes" = array<i64: 4, 32>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
    %10, %11, %12 = "transform.structured.tile"(%arg1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    "transform.yield"() : () -> ()
  }) : () -> ()
  "transform.sequence"() <{"failure_propagation_mode" = 1 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
  ^1(%arg0_1 : !transform.any_op):
    %arg1_1 = "transform.select"(%arg0_1) <{"op_name" = "linalg.quantized_matmul"}> : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
    %13, %14, %15 = "transform.structured.tile"(%arg1_1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    "transform.yield"() : () -> ()
  }) : () -> ()
}
    


//CHECK: builtin.module attributes  {"transform.with_named_sequence", "transform.sequence"} {
//CHECK-NEXT:   %0 = "test.op"() : () -> !transform.affine_map
//CHECK-NEXT:   %1 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:   %2 = "test.op"() : () -> !transform.any_param
//CHECK-NEXT:   %3 = "test.op"() : () -> !transform.any_value
//CHECK-NEXT:   %4 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
//CHECK-NEXT:   %5 = "test.op"() : () -> !transform.param<i64>
//CHECK-NEXT:   %6 = "test.op"() : () -> !transform.type
//CHECK-NEXT:   "transform.named_sequence"() <{"function_type" = (!transform.any_op, !transform.op<"linalg.quantized_matmul">, !transform.op<"linalg.elemwise_binary">) -> (), "sym_name" = "__transform_main"}> ({
//CHECK-NEXT:   ^0(%arg0 : !transform.any_op, %arg1 : !transform.op<"linalg.quantized_matmul">, %arg2 : !transform.op<"linalg.elemwise_binary">):
//CHECK-NEXT:     %7 = "transform.cast"(%arg1) : (!transform.op<"linalg.quantized_matmul">) -> !transform.any_op
//CHECK-NEXT:     %8, %9 = "transform.structured.tile_to_forall_op"(%arg1) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0, 0>, "static_tile_sizes" = array<i64: 4, 32>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
//CHECK-NEXT:     %10, %11, %12 = "transform.structured.tile"(%arg1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:     "transform.yield"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT:   "transform.sequence"() <{"failure_propagation_mode" = 1 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
//CHECK-NEXT:   ^1(%arg0_1 : !transform.any_op):
//CHECK-NEXT:     %arg1_1 = "transform.select"(%arg0_1) <{"op_name" = "linalg.quantized_matmul"}> : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
//CHECK-NEXT:     %13, %14, %15 = "transform.structured.tile"(%arg1_1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:     "transform.yield"() : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }
