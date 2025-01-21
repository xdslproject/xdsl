// RUN: xdsl-opt --print-op-generic --split-input-file %s | filecheck %s

"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"(%0, %0) : (i32, i32) -> i32
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-NEXT:   %1 = "test.op"(%0, %0) : (i32, i32) -> i32
// CHECK-NEXT: }) : () -> ()

// -----

%0 = "test.op"() : () -> i32
%1 = "test.op"(%0, %0) : (i32, i32) -> i32

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-NEXT:   %1 = "test.op"(%0, %0) : (i32, i32) -> i32
// CHECK-NEXT: }) : () -> ()

// -----

%0 = "test.op"() : () -> i32
%1 = "test.op"(%0, %0) : (i32, i32) -> i32
"builtin.module"() ({
  %2 = "test.op"() : () -> i32
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-NEXT:   %1 = "test.op"(%0, %0) : (i32, i32) -> i32
// CHECK-NEXT:   "builtin.module"() ({
// CHECK-NEXT:     %2 = "test.op"() : () -> i32
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
