// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({

  "test.op"() {name = "\""} : () -> ()
  // CHECK:      "test.op"() {"name" = "\""} : () -> ()

  "test.op"() {name = "\n"} : () -> ()
  // CHECK-NEXT: "test.op"() {"name" = "\n"} : () -> ()

  "test.op"() {name = "\t"} : () -> ()
  // CHECK-NEXT: "test.op"() {"name" = "\t"} : () -> ()

  "test.op"() {name = "\\"} : () -> ()
  // CHECK-NEXT: "test.op"() {"name" = "\\"} : () -> ()

}) : () -> ()
