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
