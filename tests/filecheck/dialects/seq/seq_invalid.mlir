// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)

  // CHECK: {{Operation does not verify: divider value 5 is not a power of 2}}
  %div_clk = seq.clock_div %clk by 5
}
