// RUN: xdsl-opt --split-input-file --parsing-diagnostics %s | filecheck %s

"test.op"() {poly_ty = !polynomial.polynomial<ring = 42 : i32>} : () -> ()

// CHECK: expected RingAttr in polynomial type, got 42 : i32
