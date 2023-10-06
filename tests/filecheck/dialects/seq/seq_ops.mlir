// RUN: XDSL_ROUNDTRIP

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)
  %div_clk = seq.clock_div %clk by 4
  // CHECK:      %div_clk = seq.clock_div %clk by 4
}
