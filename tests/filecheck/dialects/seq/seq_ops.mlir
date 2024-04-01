// RUN: XDSL_ROUNDTRIP

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)
  %data = "test.op"() : () -> i14
  %bool = "test.op"() : () -> i1

  %div_clk = seq.clock_div %clk by 4
  // CHECK:      %div_clk = seq.clock_div %clk by 4
  %compreg = seq.compreg %data, %clk : i14
  // CHECK: %compreg = seq.compreg %data, %clk : i14
  %compreg_reset = seq.compreg %data, %clk reset %bool, %data : i14
  // CHECK: %compreg_reset = seq.compreg %data, %clk reset %bool, %data : i14
  %compreg_sym = seq.compreg sym #hw<innerSym@foo> %data, %clk : i14
  // CHECK: %compreg_sym = seq.compreg sym #hw<innerSym@foo> %data, %clk : i14
}
