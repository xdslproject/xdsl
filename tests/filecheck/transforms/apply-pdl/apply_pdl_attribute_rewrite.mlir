// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

// CHECK: %val = arith.constant 1 : i32
%val = arith.constant 0 : i32

pdl.pattern : benefit(2) {
  %0 = pdl.type
  %1 = pdl.attribute = 0 : i32
  %2 = pdl.operation "arith.constant" {"value" = %1} -> (%0 : !pdl.type)
  pdl.rewrite %2 {
    %3 = pdl.attribute = 1 : i32
    %4 = pdl.operation "arith.constant" {"value" = %3} -> (%0 : !pdl.type)
    pdl.replace %2 with %4
  }
}
