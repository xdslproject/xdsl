// RUN: XDSL_ROUNDTRIP

"test.op"() {attrs = [
  !irdl_interp.variable.attr,
  // CHECK: !irdl_interp.variable.attr

  !irdl_interp.variable.range,
  // CHECK-SAME: !irdl_interp.variable.range

  !irdl_interp.variable.int,
  // CHECK-SAME: !irdl_interp.variable.int

  !irdl_interp.attr,
  // CHECK-SAME: !irdl_interp.attr

  !irdl_interp.operation
  // CHECK-SAME: !irdl_interp.operation
]}: () -> ()
