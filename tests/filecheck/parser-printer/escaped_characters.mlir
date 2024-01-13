// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

"builtin.module"() ({

  "test.op"() {name = "\""} : () -> ()
  // CHECK:      "test.op"() {"name" = "\22"} : () -> ()

  "test.op"() {name = "\n"} : () -> ()
  // CHECK-NEXT: "test.op"() {"name" = "\0A"} : () -> ()

  "test.op"() {name = "\t"} : () -> ()
  // CHECK-NEXT: "test.op"() {"name" = "\09"} : () -> ()

  "test.op"() {name = "\\"} : () -> ()
  // CHECK-NEXT: "test.op"() {"name" = "\\"} : () -> ()

}) : () -> ()
