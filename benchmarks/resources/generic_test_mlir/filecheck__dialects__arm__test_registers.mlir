"builtin.module"() ({
  "test.op"() {unallocated = !arm.reg} : () -> ()
  "test.op"() {allocated = !arm.reg<x1>} : () -> ()
}) : () -> ()
