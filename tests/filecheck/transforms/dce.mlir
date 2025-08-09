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
// CHECK-NEXT:   "test.termop"() [^bb0] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }) : () -> ()
"test.op"() ({
  "test.termop"()[^bb1] : () -> ()
^bb0:
  "test.op"() : () -> ()
  "test.termop"()[^bb1] : () -> ()
^bb1:
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
^bb0:
  %0 = "test.op"(%1) : (i32) -> (i32)
  "test.termop"()[^bb1] : () -> ()
^bb1:
  %1 = "test.op"(%0) : (i32) -> (i32)
  "test.termop"()[^bb0] : () -> ()
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
