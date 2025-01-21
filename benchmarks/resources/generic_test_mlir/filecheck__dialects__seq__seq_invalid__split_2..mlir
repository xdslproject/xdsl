"builtin.module"() ({
  %0 = "test.op"() : () -> !seq.clock
  %1 = "test.op"() : () -> i14
  %2 = "test.op"() : () -> i1
  %3 = "seq.compreg"(%1, %0, %1) {operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>} : (i14, !seq.clock, i14) -> i14
}) : () -> ()
