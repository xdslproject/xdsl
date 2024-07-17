// RUN: xdsl-opt --allow-unregistered-dialect %s -p dce | filecheck %s

/// Simple op removal
// CHECK:      "test.op"() ({
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }) : () -> ()
"test.op"() ({
  %0 = "test.pureop"() : () -> (i32)
  %1 = "test.pureop"(%0) : (i32) -> (i32)
  "test.termop"() : () -> ()
}) : () -> ()

/// Block removal
// CHECK:      "test.op"() ({
// CHECK-NEXT:   "test.termop"() [^0] : () -> ()
// CHECK-NEXT: ^0:
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }) : () -> ()
"test.op"() ({
  "test.termop"()[^1] : () -> ()
^0:
  "test.op"() : () -> ()
  "test.termop"()[^1] : () -> ()
^1:
  "test.termop"() : () -> ()
}) : () -> ()

/// Circular operation removal
// CHECK:      "test.op"() ({
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }) : () -> ()
"test.op"() ({
  %0 = "test.pureop"(%1) : (i32) -> (i32)
  %1 = "test.pureop"(%0) : (i32) -> (i32)
  "test.termop"() : () -> ()
}) : () -> ()

/// Circular block removal
// CHECK:      "test.op"() ({
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }) : () -> ()
"test.op"() ({
  "test.termop"() : () -> ()
^0:
  %0 = "test.op"(%1) : (i32) -> (i32)
  "test.termop"()[^1] : () -> ()
^1:
  %1 = "test.op"(%0) : (i32) -> (i32)
  "test.termop"()[^0] : () -> ()
}) : () -> ()

/// Recursive test
// CHECK:      "test.op"() ({
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     "test.termop"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }) : () -> ()
"test.op"() ({
  "test.op"() ({
    "test.pureop"() : () -> ()
    "test.termop"() : () -> ()
  }) : () -> ()
  "test.termop"() : () -> ()
}) : () -> ()
