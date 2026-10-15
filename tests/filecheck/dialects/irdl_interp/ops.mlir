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
