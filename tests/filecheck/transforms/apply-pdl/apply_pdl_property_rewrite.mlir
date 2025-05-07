// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

// CHECK: "test.op"() <{prop1 = i64}> : () -> ()
"test.op"() <{"prop1" = i32}> : () -> ()

pdl.pattern : benefit(42) {
  %0 = pdl.attribute = i32
  %1 = pdl.operation "test.op" {"prop1" = %0}
  pdl.rewrite %1 {
    %2 = pdl.attribute = i64
    %3 = pdl.operation "test.op" {"prop1" = %2}
    pdl.replace %1 with %3
  }
}
