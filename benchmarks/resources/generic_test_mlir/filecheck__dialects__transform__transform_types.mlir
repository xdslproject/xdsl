"builtin.module"() ({
  "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "foo"}> ({
  ^bb0(%arg4: !transform.any_op):
    "transform.yield"() : () -> ()
  }) : () -> ()
  %0 = "test.op"() : () -> !transform.affine_map
  %1 = "test.op"() : () -> !transform.any_op
  %2 = "test.op"() : () -> !transform.any_value
  %3 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
  %4 = "test.op"() : () -> !transform.param<i64>
  %5 = "test.op"() : () -> !transform.type
  "transform.named_sequence"() <{function_type = (!transform.any_op, !transform.op<"linalg.quantized_matmul">, !transform.op<"linalg.elemwise_binary">) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg1: !transform.any_op, %arg2: !transform.op<"linalg.quantized_matmul">, %arg3: !transform.op<"linalg.elemwise_binary">):
    %21 = "transform.cast"(%arg2) : (!transform.op<"linalg.quantized_matmul">) -> !transform.any_op
    %22:2 = "transform.structured.tile_using_forall"(%arg2) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, static_tile_sizes = array<i64: 4, 32>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
    %23:3 = "transform.structured.tile_using_for"(%arg2) <{scalable_sizes = array<i1: false, false>, static_sizes = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    "transform.yield"() : () -> ()
  }) : () -> ()
  "transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
  ^bb0(%arg0: !transform.any_op):
    %19 = "transform.select"(%arg0) <{op_name = "linalg.quantized_matmul"}> : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
    %20:3 = "transform.structured.tile_using_for"(%19) <{scalable_sizes = array<i1: false, false>, static_sizes = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    "transform.yield"() : () -> ()
  }) : () -> ()
  %6 = "test.op"() : () -> !transform.any_op
  %7 = "transform.get_producer_of_operand"(%6) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %8 = "transform.get_consumers_of_result"(%7) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %9 = "test.op"() : () -> !transform.any_value
  %10 = "transform.get_defining_op"(%9) : (!transform.any_value) -> !transform.any_op
  %11 = "transform.get_parent_op"(%10) <{isolated_from_above, nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op
  %12 = "transform.get_result"(%11) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
  %13 = "transform.get_type"(%12) <{elemental}> : (!transform.any_value) -> !transform.type
  "transform.include"(%11) <{failure_propagation_mode = 1 : i32, target = @foo}> : (!transform.any_op) -> ()
  "transform.match.operation_empty"(%11) : (!transform.any_op) -> ()
  "transform.match.operation_name"(%11) <{op_names = ["scf.for"]}> : (!transform.any_op) -> ()
  %14 = "transform.merge_handles"(%10, %11) : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %15 = "test.op"() : () -> !transform.any_param
  %16 = "test.op"() : () -> !transform.any_param
  "transform.match.param.cmpi"(%15, %16) <{predicate = 1 : i32}> : (!transform.any_param, !transform.any_param) -> ()
  %17:2 = "transform.split_handle"(%14) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %18 = "transform.structured.match"(%14) <{op_attrs = {qmatmul_0}}> : (!transform.any_op) -> !transform.any_op
}) {transform.with_named_sequence} : () -> ()
