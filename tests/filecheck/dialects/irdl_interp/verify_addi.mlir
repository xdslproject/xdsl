// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// Test the complete arith.addi verify function
// This demonstrates all irdl_interp operations working together

func.func @verify_addi(%op: !irdl_interp.operation) {
  // Get operand and result types
  // CHECK: %operand_types = irdl_interp.get_operand_types %op
  %operand_types = irdl_interp.get_operand_types %op
  // CHECK: %result_types = irdl_interp.get_result_types %op
  %result_types = irdl_interp.get_result_types %op

  // Check operand count == 2
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %operand_count = tensor.dim %operand_types, %[[C0]] : tensor<?x!irdl_interp.attr>
  %operand_count = tensor.dim %operand_types, %c0 : tensor<?x!irdl_interp.attr>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  %two = arith.constant 2 : index
  // CHECK: irdl_interp.unify.int %operand_count : index and %[[C2]] : index
  irdl_interp.unify.int %operand_count : index and %two : index

  // Check result count == 1
  // CHECK: %result_count = tensor.dim %result_types, %[[C0]] : tensor<?x!irdl_interp.attr>
  %result_count = tensor.dim %result_types, %c0 : tensor<?x!irdl_interp.attr>
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %one = arith.constant 1 : index
  // CHECK: irdl_interp.unify.int %result_count : index and %[[C1]] : index
  irdl_interp.unify.int %result_count : index and %one : index

  // Create constraint variable for the shared type
  // CHECK: %T = irdl_interp.new_var.attr
  %T = irdl_interp.new_var.attr

  // Extract individual operand/result types
  // CHECK: %lhs_type = tensor.extract %operand_types[%[[C0]]] : tensor<?x!irdl_interp.attr>
  %lhs_type = tensor.extract %operand_types[%c0] : tensor<?x!irdl_interp.attr>
  // CHECK: %rhs_type = tensor.extract %operand_types[%[[C0]]] : tensor<?x!irdl_interp.attr>
  %rhs_type = tensor.extract %operand_types[%c0] : tensor<?x!irdl_interp.attr>
  // CHECK: %result_type = tensor.extract %result_types[%[[C0]]] : tensor<?x!irdl_interp.attr>
  %result_type = tensor.extract %result_types[%c0] : tensor<?x!irdl_interp.attr>

  // Check each type satisfies the constraint and assert
  // CHECK: %lhs_check = irdl_interp.check_base %lhs_type : !irdl_interp.attr, "IntegerType"
  %lhs_check = irdl_interp.check_base %lhs_type : !irdl_interp.attr, "IntegerType"
  // CHECK: irdl_interp.assert %lhs_check, "Expected IntegerType for lhs"
  irdl_interp.assert %lhs_check, "Expected IntegerType for lhs"
  
  // CHECK: %rhs_check = irdl_interp.check_base %rhs_type : !irdl_interp.attr, "IntegerType"
  %rhs_check = irdl_interp.check_base %rhs_type : !irdl_interp.attr, "IntegerType"
  // CHECK: irdl_interp.assert %rhs_check, "Expected IntegerType for rhs"
  irdl_interp.assert %rhs_check, "Expected IntegerType for rhs"
  
  // CHECK: %result_check = irdl_interp.check_base %result_type : !irdl_interp.attr, "IntegerType"
  %result_check = irdl_interp.check_base %result_type : !irdl_interp.attr, "IntegerType"
  // CHECK: irdl_interp.assert %result_check, "Expected IntegerType for result"
  irdl_interp.assert %result_check, "Expected IntegerType for result"

  // Unify all types to ensure equality
  // CHECK: irdl_interp.unify.attr %lhs_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  irdl_interp.unify.attr %lhs_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  // CHECK: irdl_interp.unify.attr %rhs_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  irdl_interp.unify.attr %rhs_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  // CHECK: irdl_interp.unify.attr %result_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  irdl_interp.unify.attr %result_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr

  // Check overflowFlags property
  // CHECK: %overflow_flags = irdl_interp.get_prop %op, "overflowFlags"
  %overflow_flags = irdl_interp.get_prop %op, "overflowFlags"
  // CHECK: %overflow_check = irdl_interp.check_base %overflow_flags : !irdl_interp.attr, "IntegerOverflowAttr"
  %overflow_check = irdl_interp.check_base %overflow_flags : !irdl_interp.attr, "IntegerOverflowAttr"
  // CHECK: irdl_interp.assert %overflow_check, "Expected IntegerOverflowAttr"
  irdl_interp.assert %overflow_check, "Expected IntegerOverflowAttr"

  // Check there are no extra properties (exactly 1 property expected)
  // CHECK: %prop_count = irdl_interp.get_prop_count %op
  %prop_count = irdl_interp.get_prop_count %op
  // CHECK: irdl_interp.unify.int %prop_count : index and %one : index
  irdl_interp.unify.int %prop_count : index and %one : index

  func.return
}

// CHECK-GENERIC:      "func.func"() <{sym_name = "verify_addi", function_type = (!irdl_interp.operation) -> ()}> ({
// CHECK-GENERIC-NEXT: ^bb0(%op: !irdl_interp.operation):
// CHECK-GENERIC-NEXT:   %operand_types = "irdl_interp.get_operand_types"(%op) : (!irdl_interp.operation) -> tensor<?x!irdl_interp.attr>
// CHECK-GENERIC-NEXT:   %result_types = "irdl_interp.get_result_types"(%op) : (!irdl_interp.operation) -> tensor<?x!irdl_interp.attr>
// CHECK-GENERIC-NEXT:   %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-GENERIC-NEXT:   %operand_count = "tensor.dim"(%operand_types, %c0) : (tensor<?x!irdl_interp.attr>, index) -> index
// CHECK-GENERIC-NEXT:   %two = "arith.constant"() <{value = 2 : index}> : () -> index
// CHECK-GENERIC-NEXT:   "irdl_interp.unify.int"(%operand_count, %two) : (index, index) -> ()
// CHECK-GENERIC-NEXT:   %result_count = "tensor.dim"(%result_types, %c0) : (tensor<?x!irdl_interp.attr>, index) -> index
// CHECK-GENERIC-NEXT:   %one = "arith.constant"() <{value = 1 : index}> : () -> index
// CHECK-GENERIC-NEXT:   "irdl_interp.unify.int"(%result_count, %one) : (index, index) -> ()
// CHECK-GENERIC-NEXT:   %T = "irdl_interp.new_var.attr"() : () -> !irdl_interp.variable.attr
// CHECK-GENERIC-NEXT:   %lhs_type = "tensor.extract"(%operand_types, %c0) : (tensor<?x!irdl_interp.attr>, index) -> !irdl_interp.attr
// CHECK-GENERIC-NEXT:   %rhs_type = "tensor.extract"(%operand_types, %c0) : (tensor<?x!irdl_interp.attr>, index) -> !irdl_interp.attr
// CHECK-GENERIC-NEXT:   %result_type = "tensor.extract"(%result_types, %c0) : (tensor<?x!irdl_interp.attr>, index) -> !irdl_interp.attr
// CHECK-GENERIC-NEXT:   %lhs_check = "irdl_interp.check_base"(%lhs_type) <{base = "IntegerType"}> : (!irdl_interp.attr) -> i1
// CHECK-GENERIC-NEXT:   "irdl_interp.assert"(%lhs_check) <{message = "Expected IntegerType for lhs"}> : (i1) -> ()
// CHECK-GENERIC-NEXT:   %rhs_check = "irdl_interp.check_base"(%rhs_type) <{base = "IntegerType"}> : (!irdl_interp.attr) -> i1
// CHECK-GENERIC-NEXT:   "irdl_interp.assert"(%rhs_check) <{message = "Expected IntegerType for rhs"}> : (i1) -> ()
// CHECK-GENERIC-NEXT:   %result_check = "irdl_interp.check_base"(%result_type) <{base = "IntegerType"}> : (!irdl_interp.attr) -> i1
// CHECK-GENERIC-NEXT:   "irdl_interp.assert"(%result_check) <{message = "Expected IntegerType for result"}> : (i1) -> ()
// CHECK-GENERIC-NEXT:   "irdl_interp.unify.attr"(%lhs_type, %T) : (!irdl_interp.attr, !irdl_interp.variable.attr) -> ()
// CHECK-GENERIC-NEXT:   "irdl_interp.unify.attr"(%rhs_type, %T) : (!irdl_interp.attr, !irdl_interp.variable.attr) -> ()
// CHECK-GENERIC-NEXT:   "irdl_interp.unify.attr"(%result_type, %T) : (!irdl_interp.attr, !irdl_interp.variable.attr) -> ()
// CHECK-GENERIC-NEXT:   %overflow_flags = "irdl_interp.get_prop"(%op) <{prop_name = "overflowFlags"}> : (!irdl_interp.operation) -> !irdl_interp.attr
// CHECK-GENERIC-NEXT:   %overflow_check = "irdl_interp.check_base"(%overflow_flags) <{base = "IntegerOverflowAttr"}> : (!irdl_interp.attr) -> i1
// CHECK-GENERIC-NEXT:   "irdl_interp.assert"(%overflow_check) <{message = "Expected IntegerOverflowAttr"}> : (i1) -> ()
// CHECK-GENERIC-NEXT:   %prop_count = "irdl_interp.get_prop_count"(%op) : (!irdl_interp.operation) -> index
// CHECK-GENERIC-NEXT:   "irdl_interp.unify.int"(%prop_count, %one) : (index, index) -> ()
// CHECK-GENERIC-NEXT:   "func.return"() : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
