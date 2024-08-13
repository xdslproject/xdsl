builtin.module attributes  {"transform.with_named_sequence"} {
  "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "foo"}> ({
  ^bb0(%arg0: !transform.any_op):
    "transform.yield"() : () -> ()
  }) : () -> ()
  %16 = "test.op"() : () -> !transform.any_op
  %17 = "transform.get_producer_of_operand"(%16) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %18 = "transform.get_consumers_of_result"(%17) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %19 = "test.op"() : () -> !transform.any_value
  %20 = "transform.get_defining_op" (%19) : (!transform.any_value) -> !transform.any_op
  %21 = "transform.get_parent_op"(%20) <{isolated_from_above, nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op
  %22 = "transform.get_result"(%21) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_value
  %23 = "transform.get_type"(%22) <{elemental}> : (!transform.any_value) -> !transform.type
  "transform.include"(%21) <{failure_propagation_mode = 1 : i32, target = @foo}> : (!transform.any_op) -> ()
  "transform.match.operation_empty"(%21) : (!transform.any_op) -> ()
  "transform.match.operation_name" (%21) <{op_names = ["scf.for"]}> : (!transform.any_op) -> ()
  %24 = "transform.merge_handles"(%20, %21) : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %25 = "test.op"() : () -> !transform.any_param
  %26 = "test.op"() : () -> !transform.any_param
  "transform.match.param.cmpi"(%25, %26) <{predicate = 1 : i32}> : (!transform.any_param, !transform.any_param) -> ()
  %27:2 = "transform.split_handle"(%24) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)}
