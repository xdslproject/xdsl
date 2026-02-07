// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

builtin.module attributes  {"transform.with_named_sequence"} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  %0 = "test.op"() : () -> !transform.affine_map
  %1 = "test.op"() : () -> !transform.any_op
  %3 = "test.op"() : () -> !transform.any_value
  %4 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
  %5 = "test.op"() : () -> !transform.param<i64>
  %6 = "test.op"() : () -> !transform.type
  transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.quantized_matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
    %7 = "transform.cast"(%arg1) : (!transform.op<"linalg.quantized_matmul">) -> !transform.any_op
    %8, %9 = "transform.structured.tile_using_forall"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, "static_tile_sizes" = array<i64: 4, 32>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
    %10, %11, %12 = "transform.structured.tile_using_for"(%arg1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
  "transform.sequence"() <{"failure_propagation_mode" = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
  ^bb1(%arg0_1 : !transform.any_op):
    %arg1_1 = "transform.select"(%arg0_1) <{"op_name" = "linalg.quantized_matmul"}> : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
    %13, %14, %15 = "transform.structured.tile_using_for"(%arg1_1) <{"scalable_sizes" = array<i1: false, false>, "static_sizes" = array<i64: 8, 8>}> : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }) : () -> ()
 %16 = "test.op"() : () -> !transform.any_op
  %17 = "transform.get_producer_of_operand"(%16) <{operand_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %18 = "transform.get_consumers_of_result"(%17) <{result_number = 0 : i64}> : (!transform.any_op) -> !transform.any_op
  %19 = "test.op"() : () -> !transform.any_value
  %20 = "transform.get_defining_op" (%19) : (!transform.any_value) -> !transform.any_op
  %21 = "transform.get_parent_op"(%20) <{isolated_from_above, nth_parent = 1 : i64}> : (!transform.any_op) -> !transform.any_op
  %22 = "transform.get_result"(%21) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
  %23 = "transform.get_type"(%22) <{elemental}> : (!transform.any_value) -> !transform.type
  "transform.include"(%21) <{failure_propagation_mode = 1 : i32, target = @foo}> : (!transform.any_op) -> ()
  "transform.match.operation_empty"(%21) : (!transform.any_op) -> ()
  "transform.match.operation_name" (%21) <{op_names = ["scf.for"]}> : (!transform.any_op) -> ()
  %24 = "transform.merge_handles"(%20, %21) : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %25 = "test.op"() : () -> !transform.any_param
  %26 = "test.op"() : () -> !transform.any_param
  "transform.match.param.cmpi"(%25, %26) <{predicate = 1 : i32}> : (!transform.any_param, !transform.any_param) -> ()
  %27:2 = "transform.split_handle"(%24) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %28 = "transform.structured.match"(%24) <{"op_attrs" = {"qmatmul_0"}}> : (!transform.any_op) -> !transform.any_op
  %29 = "test.op"() : () -> !transform.op<"builtin.module">
  %30 = transform.apply_registered_pass "foo" to %29 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
  %31 = transform.apply_registered_pass "foo" with options = {foo = 1 : i32} to %30 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
}



//CHECK: module attributes {transform.with_named_sequence} {
//CHECK-NEXT:   transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) {
//CHECK-NEXT:     transform.yield
//CHECK-NEXT:   }
//CHECK-NEXT:   %0 = "test.op"() : () -> !transform.affine_map
//CHECK-NEXT:   %1 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:   %2 = "test.op"() : () -> !transform.any_value
//CHECK-NEXT:   %3 = "test.op"() : () -> !transform.op<"linalg.quantized_matmul">
//CHECK-NEXT:   %4 = "test.op"() : () -> !transform.param<i64>
//CHECK-NEXT:   %5 = "test.op"() : () -> !transform.type
//CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.quantized_matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
//CHECK-NEXT:     %22 = transform.cast %arg1 : !transform.op<"linalg.quantized_matmul"> to !transform.any_op
//CHECK-NEXT:     %tiled_op, %forall_op = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32] : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op)
//CHECK-NEXT:     %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %arg1 tile_sizes [8, 8] : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:     transform.yield
//CHECK-NEXT:   }
//CHECK-NEXT:   transform.sequence  failures(propagate) {
//CHECK-NEXT:   ^bb0(%arg0: !transform.any_op):
//CHECK-NEXT:     %22 = select "linalg.quantized_matmul" in %arg0 : (!transform.any_op) -> !transform.op<"linalg.quantized_matmul">
//CHECK-NEXT:     %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %22 tile_sizes [8, 8] : (!transform.op<"linalg.quantized_matmul">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//CHECK-NEXT:   }
//CHECK-NEXT:   %6 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:   %7 = transform.get_producer_of_operand %6[0] : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:   %8 = transform.get_consumers_of_result %7[0] : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:   %9 = "test.op"() : () -> !transform.any_value
//CHECK-NEXT:   %10 = transform.get_defining_op %9 : (!transform.any_value) -> !transform.any_op
//CHECK-NEXT:   %11 = transform.get_parent_op %10 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:   %12 = transform.get_result %11[0] : (!transform.any_op) -> !transform.any_value
//CHECK-NEXT:   %13 = transform.get_type elemental %12 : (!transform.any_value) -> !transform.type
//CHECK-NEXT:   transform.include @foo failures(propagate) (%11) : (!transform.any_op) -> ()
//CHECK-NEXT:   transform.match.operation_empty %11 : !transform.any_op
//CHECK-NEXT:   transform.match.operation_name %11 ["scf.for"] : !transform.any_op
//CHECK-NEXT:   %14 = transform.merge_handles %10, %11 : !transform.any_op
//CHECK-NEXT:   %15 = "test.op"() : () -> !transform.any_param
//CHECK-NEXT:   %16 = "test.op"() : () -> !transform.any_param
//CHECK-NEXT:   transform.match.param.cmpi ne %15, %16 : !transform.any_param
//CHECK-NEXT:   %17:2 = transform.split_handle %14 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//CHECK-NEXT:   %18 = transform.structured.match attributes {qmatmul_0} in %14 : (!transform.any_op) -> !transform.any_op
//CHECK-NEXT:   %19 = "test.op"() : () -> !transform.op<"builtin.module">
//CHECK-NEXT:   %20 = transform.apply_registered_pass "foo" to %19 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
//CHECK-NEXT:   %21 = transform.apply_registered_pass "foo" with options = {"foo" = 1 : i32} to %20 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
//CHECK-NEXT: }
