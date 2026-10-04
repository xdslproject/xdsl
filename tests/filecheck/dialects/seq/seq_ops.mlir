// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)
  %data = "test.op"() : () -> i14
  %on = "test.op"() : () -> i14
  %bool = "test.op"() : () -> i1

  %div_clk = seq.clock_div %clk by 4
  // CHECK:      %div_clk = seq.clock_div %clk by 4
  // CHECK-GENERIC: %div_clk = "seq.clock_div"(%clk) {pow2 = 4 : i8} : (!seq.clock) -> !seq.clock
  %div_clk_attr = seq.clock_div %clk by 2 {foo}
  // CHECK: %div_clk_attr = seq.clock_div %clk by 2 {foo}
  // CHECK-GENERIC: %div_clk_attr = "seq.clock_div"(%clk) {pow2 = 2 : i8, foo} : (!seq.clock) -> !seq.clock
  %compreg = seq.compreg %data, %clk : i14
  // CHECK: %compreg = seq.compreg %data, %clk : i14
  %compreg_reset = seq.compreg %data, %clk reset %bool, %data : i14
  // CHECK: %compreg_reset = seq.compreg %data, %clk reset %bool, %data : i14
  %compreg_poweron = seq.compreg %data, %clk powerOn %on : i14
  // CHECK: %compreg_poweron = seq.compreg %data, %clk powerOn %on : i14
  %compreg_all = seq.compreg %data, %clk reset %bool, %data powerOn %on : i14
  // CHECK: %compreg_all = seq.compreg %data, %clk reset %bool, %data powerOn %on : i14
  %compreg_sym = seq.compreg sym @foo %data, %clk : i14
  // CHECK: %compreg_sym = seq.compreg sym @foo %data, %clk : i14
  // CHECK-GENERIC:    %compreg_sym = "seq.compreg"(%data, %clk) {inner_sym = #hw<innerSym@foo>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>} : (i14, !seq.clock) -> i14

  %const_low = seq.const_clock low
  // CHECK: %const_low = seq.const_clock low
  %const_high = seq.const_clock high
  // CHECK: %const_high = seq.const_clock high
  %const_low_foo = seq.const_clock low {"foo"}
  // CHECK: %const_low_foo = seq.const_clock low {foo}
}
