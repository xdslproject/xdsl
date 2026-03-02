// RUN: XDSL_ROUNDTRIP

// CHECK: %qmatmul = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
%qmatmul = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">

// CHECK: %param = "test.op"() : () -> !transform.param<!transform.any_value>
%param = "test.op"() : () -> !transform.param<!transform.any_value>

// CHECK: %any_op = "test.op"() : () -> !transform.any_op
%any_op = "test.op"() : () -> !transform.any_op

// CHECK: %any_value = "test.op"() : () -> !transform.any_value
%any_value = "test.op"() : () -> !transform.any_value

// CHECK: %affine_map = "test.op"() : () -> !transform.affine_map
%affine_map = "test.op"() : () -> !transform.affine_map

// CHECK: "transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
// CHECK: ^bb0(%arg0 : !transform.any_value, %arg1 : !transform.op<"linalg.matmul">):
// CHECK:   transform.yield
// CHECK: }) : () -> ()
"transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
^bb0(%arg0 : !transform.any_value, %arg1 : !transform.op<"linalg.matmul">):
  transform.yield
}) : () -> ()

%input = "test.op"() : () -> !transform.any_value
// CHECK: %tiled_op, %loop_op, %remainder = "transform.structured.tile_using_for"(%input) <{static_sizes = array<i64: 8, 8>}> : (!transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op)
%tiled_op, %loop_op, %remainder = "transform.structured.tile_using_for"(%input) <{static_sizes = array<i64: 8, 8>}> : (!transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op)

%producer = "test.op"() : () -> !transform.any_op
// CHECK: %consumers = "transform.get_consumers_of_result"(%producer) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
%consumers = "transform.get_consumers_of_result"(%producer) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op

%value = "test.op"() : () -> !transform.any_value
// CHECK: %defining_op = "transform.get_defining_op"(%value) : (!transform.any_value) -> !transform.any_op
%defining_op = "transform.get_defining_op"(%value) : (!transform.any_value) -> !transform.any_op

%child = "test.op"() : () -> !transform.any_op
// CHECK: %parent = "transform.get_parent_op"(%child) <{nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op
%parent = "transform.get_parent_op"(%child) <{nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op

%operand_op = "test.op"() : () -> !transform.any_op
// CHECK: %producer2 = "transform.get_producer_of_operand"(%operand_op) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
%producer2 = "transform.get_producer_of_operand"(%operand_op) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op

%op = "test.op"() : () -> !transform.any_op
// CHECK: %result = "transform.get_result"(%op) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
%result = "transform.get_result"(%op) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value

%value_1 = "test.op"() : () -> !transform.any_value
// CHECK: %type = "transform.get_type"(%value_1) : (!transform.any_value) -> !transform.type
%type = "transform.get_type"(%value_1) : (!transform.any_value) -> !transform.type
// CHECK: %type_1 = "transform.get_type"(%value_1) <{elemental}> : (!transform.any_value) -> !transform.type
%type_1 = "transform.get_type"(%value_1) <{elemental}> : (!transform.any_value) -> !transform.type

%target = "test.op"() : () -> !transform.any_value
// CHECK: %included = "transform.include"(%target) <{target = @foo, failure_propagation_mode = false}> : (!transform.any_value) -> !transform.any_value
%included = "transform.include"(%target) <{target = @foo, failure_propagation_mode = false}> : (!transform.any_value) -> !transform.any_value

%empty_op = "test.op"() : () -> !transform.any_op
// CHECK: "transform.match.operation_empty"(%empty_op) : (!transform.any_op) -> ()
"transform.match.operation_empty"(%empty_op) : (!transform.any_op) -> ()

%named_op = "test.op"() : () -> !transform.any_op
// CHECK: "transform.match.operation_name"(%named_op) <{op_names = ["foo"]}> : (!transform.any_op) -> ()
"transform.match.operation_name"(%named_op) <{op_names = ["foo"]}> : (!transform.any_op) -> ()

%param1 = "test.op"() : () -> !transform.any_param
%param2 = "test.op"() : () -> !transform.any_param
// CHECK: "transform.match.param.cmpi"(%param1, %param2) <{predicate = 0 : i64}> : (!transform.any_param, !transform.any_param) -> ()
"transform.match.param.cmpi"(%param1, %param2) <{predicate = 0 : i64}> : (!transform.any_param, !transform.any_param) -> ()

%handle = "test.op"() : () -> !transform.any_op
// CHECK: %merged = "transform.merge_handles"(%handle) <{deduplicate}> : (!transform.any_op) -> !transform.any_op
%merged = "transform.merge_handles"(%handle) <{deduplicate}> : (!transform.any_op) -> !transform.any_op

// CHECK: %const_param = "transform.param.constant"() <{value = 1 : i32}> : () -> !transform.param<i32>
%const_param = "transform.param.constant"() <{value = 1 : i32}> : () -> !transform.param<i32>

%handle_to_split = "test.op"() : () -> !transform.any_op
// CHECK: %split1, %split2 = "transform.split_handle"(%handle_to_split) <{pass_through_empty_handle = true, fail_on_payload_too_small = true, overflow_result = 1 : i64}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
%split1, %split2 = "transform.split_handle"(%handle_to_split) <{pass_through_empty_handle = true, fail_on_payload_too_small = true, overflow_result = 1 : i64}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

%to_tile = "test.op"() : () -> !transform.any_value
// CHECK: %tiled, %loop = "transform.structured.tile_using_for"(%to_tile) <{static_sizes = array<i64: 8, 0>}> : (!transform.any_value) -> (!transform.any_op, !transform.any_op)
%tiled, %loop = "transform.structured.tile_using_for"(%to_tile) <{static_sizes = array<i64: 8, 0>}> : (!transform.any_value) -> (!transform.any_op, !transform.any_op)

%to_match = "test.op"() : () -> !transform.any_op
// CHECK: %matched = "transform.structured.match"(%to_match) <{ops = [], op_attrs = {}}> : (!transform.any_op) -> !transform.any_op
%matched = "transform.structured.match"(%to_match) <{ops = [], op_attrs = {}}> : (!transform.any_op) -> !transform.any_op

%to_apply_registered_pass = "test.op"() : () -> !transform.op<"builtin.module">
// CHECK: %applied_registered_pass = transform.apply_registered_pass "foo" to %to_apply_registered_pass : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
%applied_registered_pass = transform.apply_registered_pass "foo" to %to_apply_registered_pass : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">

// CHECK: %applied_registered_pass_opts = transform.apply_registered_pass "foo" with options = {foo = 1 : i32} to %to_apply_registered_pass : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
%applied_registered_pass_opts = transform.apply_registered_pass "foo" with options = {foo = 1 : i32} to %to_apply_registered_pass : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
