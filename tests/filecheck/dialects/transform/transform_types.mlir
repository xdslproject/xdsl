// RUN: XDSL_ROUNDTRIP

builtin.module attributes  {"transform.with_named_sequence"} {
  transform.named_sequence @foo(%arg0 : !transform.any_op {transform.readonly}) {
    transform.yield
  }
  %0 = "test.op"() : () -> !transform.affine_map
  %1 = "test.op"() : () -> !transform.any_op
  %2 = "test.op"() : () -> !transform.any_value
  %3 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
  %4 = "test.op"() : () -> !transform.param<i64>
  %5 = "test.op"() : () -> !transform.type
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op, %arg1 : !transform.op<"linalg.quantized_matmul">, %arg2 : !transform.op<"linalg.elemwise_binary">) {
    %6 = "transform.cast"(%arg1) : (!transform.op<"linalg.quantized_matmul">) -> !transform.any_op
    %7, %8 = "transform.structured.tile_using_forall"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, "static_tile_sizes" = array<i64: 4, 32>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
    %9, %10, %11 = "transform.structured.tile_using_for"(%arg1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  "transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%arg0 : !transform.any_op):
      %arg1 = "transform.select"(%arg0) <{"op_name" = "linalg.quantized_matmul"}> : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
      %6, %7, %8 = "transform.structured.tile_using_for"(%arg1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }) : () -> ()
  %6 = "test.op"() : () -> !transform.any_op
  %7 = "transform.get_producer_of_operand"(%6) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %8 = "transform.get_consumers_of_result"(%7) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %9 = "test.op"() : () -> !transform.any_value
  %10 = "transform.get_defining_op" (%9) : (!transform.any_value) -> !transform.any_op
  %11 = "transform.get_parent_op"(%10) <{isolated_from_above, nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op
  %12 = "transform.get_result"(%11) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
  %13 = "transform.get_type"(%12) <{elemental}> : (!transform.any_value) -> !transform.type
  "transform.include"(%11) <{failure_propagation_mode = 1 : i32, target = @foo}> : (!transform.any_op) -> ()
  "transform.match.operation_empty"(%11) : (!transform.any_op) -> ()
  "transform.match.operation_name" (%11) <{op_names = ["scf.for"]}> : (!transform.any_op) -> ()
  %14 = "transform.merge_handles"(%10, %11) : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %15 = "test.op"() : () -> !transform.any_param
  %16 = "test.op"() : () -> !transform.any_param
  "transform.match.param.cmpi"(%15, %16) <{predicate = 1 : i32}> : (!transform.any_param, !transform.any_param) -> ()
  %17:2 = "transform.split_handle"(%14) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %18 = "transform.structured.match"(%14) <{"op_attrs" = {"qmatmul_0"}}> : (!transform.any_op) -> !transform.any_op
}



//CHECK: builtin.module attributes  {transform.with_named_sequence} {
//CHECK-NEXT:  transform.named_sequence @foo(%arg0 : !transform.any_op {transform.readonly}) {
//CHECK-NEXT:    transform.yield
//CHECK-NEXT:  }
//CHECK-NEXT:  %0 = "test.op"() : () -> !transform.affine_map
//CHECK-NEXT:  %1 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:  %2 = "test.op"() : () -> !transform.any_value
//CHECK-NEXT:  %3 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
//CHECK-NEXT:  %4 = "test.op"() : () -> !transform.param<i64>
//CHECK-NEXT:  %5 = "test.op"() : () -> !transform.type
//CHECK-NEXT:  transform.named_sequence @__transform_main(%arg0 : !transform.any_op, %arg1 : !transform.op<"linalg.quantized_matmul">, %arg2 : !transform.op<"linalg.elemwise_binary">) {
//CHECK-NEXT:    %6 = "transform.cast"(%arg1) : (!transform.op<"linalg.quantized_matmul">) -> !transform.any_op
//CHECK-NEXT:    %7, %8 = "transform.structured.tile_using_forall"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, static_tile_sizes = array<i64: 4, 32>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
//CHECK-NEXT:    %9, %10, %11 = "transform.structured.tile_using_for"(%arg1) <{scalable_sizes = array<i1: false, false>, static_sizes = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:    transform.yield
//CHECK-NEXT:  }
//CHECK-NEXT:  "transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
//CHECK-NEXT:  ^bb0(%arg0 : !transform.any_op):
//CHECK-NEXT:    %arg1 = "transform.select"(%arg0) <{op_name = "linalg.quantized_matmul"}> : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
//CHECK-NEXT:    %6, %7, %8 = "transform.structured.tile_using_for"(%arg1) <{scalable_sizes = array<i1: false, false>, static_sizes = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:    transform.yield
//CHECK-NEXT:  }) : () -> ()
//CHECK-NEXT:  %6 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:  %7 = "transform.get_producer_of_operand"(%6) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:  %8 = "transform.get_consumers_of_result"(%7) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:  %9 = "test.op"() : () -> !transform.any_value
//CHECK-NEXT:  %10 = "transform.get_defining_op"(%9) : (!transform.any_value) -> !transform.any_op
//CHECK-NEXT:  %11 = "transform.get_parent_op"(%10) <{isolated_from_above, nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:  %12 = "transform.get_result"(%11) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
//CHECK-NEXT:  %13 = "transform.get_type"(%12) <{elemental}> : (!transform.any_value) -> !transform.type
//CHECK-NEXT:  "transform.include"(%11) <{failure_propagation_mode = 1 : i32, target = @foo}> : (!transform.any_op) -> ()
//CHECK-NEXT:  "transform.match.operation_empty"(%11) : (!transform.any_op) -> ()
//CHECK-NEXT:  "transform.match.operation_name"(%11) <{op_names = ["scf.for"]}> : (!transform.any_op) -> ()
//CHECK-NEXT:  %14 = "transform.merge_handles"(%10, %11) : (!transform.any_op, !transform.any_op) -> !transform.any_op
//CHECK-NEXT:  %15 = "test.op"() : () -> !transform.any_param
//CHECK-NEXT:  %16 = "test.op"() : () -> !transform.any_param
//CHECK-NEXT:  "transform.match.param.cmpi"(%15, %16) <{predicate = 1 : i32}> : (!transform.any_param, !transform.any_param) -> ()
//CHECK-NEXT:  %17, %18 = "transform.split_handle"(%14) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//CHECK-NEXT:  %19 = "transform.structured.match"(%14) <{op_attrs = {qmatmul_0}}> : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:}
