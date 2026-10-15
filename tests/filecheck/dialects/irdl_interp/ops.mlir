// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: %var_attr = irdl_interp.new_var.attr
// CHECK-GENERIC: %var_attr = "irdl_interp.new_var.attr"() : () -> !irdl_interp.variable.attr
%var_attr = irdl_interp.new_var.attr

// CHECK: %var_range = irdl_interp.new_var.range
// CHECK-GENERIC: %var_range = "irdl_interp.new_var.range"() : () -> !irdl_interp.variable.range
%var_range = irdl_interp.new_var.range

// CHECK: %var_int = irdl_interp.new_var.int
// CHECK-GENERIC: %var_int = "irdl_interp.new_var.int"() : () -> !irdl_interp.variable.int
%var_int = irdl_interp.new_var.int

// CHECK: %attr = irdl_interp.constant.attr i32
// CHECK-GENERIC: %attr = "irdl_interp.constant.attr"() <{value = i32}> : () -> !irdl_interp.attr
%attr = irdl_interp.constant.attr i32

// CHECK: %attr2 = irdl_interp.constant.attr true
// CHECK-GENERIC: %attr2 = "irdl_interp.constant.attr"() <{value = true}> : () -> !irdl_interp.attr
%attr2 = irdl_interp.constant.attr true

// CHECK: irdl_interp.unify.attr %attr : !irdl_interp.attr and %var_attr : !irdl_interp.variable.attr
// CHECK-GENERIC: "irdl_interp.unify.attr"(%attr, %var_attr) : (!irdl_interp.attr, !irdl_interp.variable.attr) -> ()
irdl_interp.unify.attr %attr : !irdl_interp.attr and %var_attr : !irdl_interp.variable.attr

// CHECK: irdl_interp.unify.attr %attr2 : !irdl_interp.attr and %attr : !irdl_interp.attr
// CHECK-GENERIC: "irdl_interp.unify.attr"(%attr2, %attr) : (!irdl_interp.attr, !irdl_interp.attr) -> ()
irdl_interp.unify.attr %attr2 : !irdl_interp.attr and %attr : !irdl_interp.attr

// CHECK: %int_val = arith.constant 42 : index
// CHECK-GENERIC: %int_val = "arith.constant"() <{value = 42 : index}> : () -> index
%int_val = arith.constant 42 : index

// CHECK: irdl_interp.unify.int %int_val : index and %var_int : !irdl_interp.variable.int
// CHECK-GENERIC: "irdl_interp.unify.int"(%int_val, %var_int) : (index, !irdl_interp.variable.int) -> ()
irdl_interp.unify.int %int_val : index and %var_int : !irdl_interp.variable.int

// CHECK: %int_val2 = arith.constant 42 : index
// CHECK-GENERIC: %int_val2 = "arith.constant"() <{value = 42 : index}> : () -> index
%int_val2 = arith.constant 42 : index

// CHECK: irdl_interp.unify.int %int_val : index and %int_val2 : index
// CHECK-GENERIC: "irdl_interp.unify.int"(%int_val, %int_val2) : (index, index) -> ()
irdl_interp.unify.int %int_val : index and %int_val2 : index

// CHECK: %range = tensor.from_elements %attr, %attr2 : tensor<2x!irdl_interp.attr>
// CHECK-GENERIC: %range = "tensor.from_elements"(%attr, %attr2) : (!irdl_interp.attr, !irdl_interp.attr) -> tensor<2x!irdl_interp.attr>
%range = tensor.from_elements %attr, %attr2 : tensor<2x!irdl_interp.attr>

// CHECK: irdl_interp.unify.range %range : tensor<2x!irdl_interp.attr> and %var_range : !irdl_interp.variable.range
// CHECK-GENERIC: "irdl_interp.unify.range"(%range, %var_range) : (tensor<2x!irdl_interp.attr>, !irdl_interp.variable.range) -> ()
irdl_interp.unify.range %range : tensor<2x!irdl_interp.attr> and %var_range : !irdl_interp.variable.range

// CHECK: %range2 = tensor.from_elements %attr, %attr2 : tensor<2x!irdl_interp.attr>
// CHECK-GENERIC: %range2 = "tensor.from_elements"(%attr, %attr2) : (!irdl_interp.attr, !irdl_interp.attr) -> tensor<2x!irdl_interp.attr>
%range2 = tensor.from_elements %attr, %attr2 : tensor<2x!irdl_interp.attr>

// CHECK: irdl_interp.unify.range %range : tensor<2x!irdl_interp.attr> and %range2 : tensor<2x!irdl_interp.attr>
// CHECK-GENERIC: "irdl_interp.unify.range"(%range, %range2) : (tensor<2x!irdl_interp.attr>, tensor<2x!irdl_interp.attr>) -> ()
irdl_interp.unify.range %range : tensor<2x!irdl_interp.attr> and %range2 : tensor<2x!irdl_interp.attr>

// CHECK: irdl_interp.check_base %attr : !irdl_interp.attr, "IntegerType"
// CHECK-GENERIC: "irdl_interp.check_base"(%attr) <{base = "IntegerType"}> : (!irdl_interp.attr) -> ()
irdl_interp.check_base %attr : !irdl_interp.attr, "IntegerType"

// CHECK: irdl_interp.check_base %var_attr : !irdl_interp.variable.attr, "IntegerType"
// CHECK-GENERIC: "irdl_interp.check_base"(%var_attr) <{base = "IntegerType"}> : (!irdl_interp.variable.attr) -> ()
irdl_interp.check_base %var_attr : !irdl_interp.variable.attr, "IntegerType"
