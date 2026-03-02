// RUN: xdsl-opt %s --syntax-highlight | filecheck %s

// CHECK: [95m%0[0m = "test.op"() : () -> i32
%0 = "test.op"() : () -> i32

// CHECK: "test.op"([95m%0[0m) : (i32) -> ()
"test.op"(%0) : (i32) -> ()
