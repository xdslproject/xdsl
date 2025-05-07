// RUN: xdsl-opt %s | filecheck %s
// RUN: mlir-opt %s --allow-unregistered-dialect | filecheck %s

// CHECK: module

"test.op"() {"name"} : () -> ()
// CHECK-NEXT: {name}

"test.op"() {"_name"} : () -> ()
// CHECK-NEXT: {_name}

"test.op"() {"Name"} : () -> ()
// CHECK-NEXT: {Name}

"test.op"() {"name$"} : () -> ()
// CHECK-NEXT: {name$}

"test.op"() {"name_"} : () -> ()
// CHECK-NEXT: {name_}

"test.op"() {"name."} : () -> ()
// CHECK-NEXT: {name.}

"test.op"() {"name0"} : () -> ()
// CHECK-NEXT: {name0}

"test.op"() {"name%"} : () -> ()
// CHECK-NEXT: {"name%"}

"test.op"() {"name#"} : () -> ()
// CHECK-NEXT: {"name#"}

"test.op"() {"name-"} : () -> ()
// CHECK-NEXT: {"name-"}

"test.op"() {"name{"} : () -> ()
// CHECK-NEXT: {"name{"}

"test.op"() {"name("} : () -> ()
// CHECK-NEXT: {"name("}

"test.op"() {"0name"} : () -> ()
// CHECK-NEXT: {"0name"}

"test.op"() {"name" = "name"} : () -> ()
// CHECK-NEXT: {name = "name"}

"test.op"() {"name#" = "name#"} : () -> ()
// CHECK-NEXT: {"name#" = "name#"}

"test.op"() {"name0" = "name0"} : () -> ()
// CHECK-NEXT: {name0 = "name0"}
