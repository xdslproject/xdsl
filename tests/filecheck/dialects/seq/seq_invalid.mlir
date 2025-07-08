// RUN: xdsl-opt --verify-diagnostics --parsing-diagnostics --split-input-file %s | filecheck %s

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)

  // CHECK: {{Operation does not verify: divider value 5 is not a power of 2}}
  %div_clk = seq.clock_div %clk by 5
}

// -----

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)
  %data = "test.op"() : () -> i14
  %bool = "test.op"() : () -> i1

  // CHECK: Both reset and reset_value must be set when one is
  %compreg_reset = "seq.compreg"(%data, %clk, %bool) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>} : (i14, !seq.clock, i1) -> i14
}

// -----

builtin.module {
  %clk = "test.op"() : () -> (!seq.clock)
  %data = "test.op"() : () -> i14
  %bool = "test.op"() : () -> i1

  // CHECK: Both reset and reset_value must be set when one is
  %compreg_reset = "seq.compreg"(%data, %clk, %data) {operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>} : (i14, !seq.clock, i14) -> i14
}

// -----

builtin.module {
  // CHECK: Expected either low or high clock value
  %clock = seq.const_clock foobar
}
