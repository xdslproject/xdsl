// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

// CHECK-NOT:  "test.op"()
"test.op"() : () -> ()

pdl.pattern : benefit(42) {
  %0 = pdl.operation "test.op"
  pdl.rewrite %0 {
    pdl.erase %0
  }
}
