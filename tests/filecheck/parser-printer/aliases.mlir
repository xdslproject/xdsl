// RUN: xdsl-opt %s --split-input-file --parsing-diagnostics | filecheck %s

!ii32 = i32
#attr = 0 : !ii32

"test.op"() {"attr" = 1 : !ii32} : () -> ()
"test.op"() {"attr" = !ii32} : () -> ()
"test.op"() {"attr" = vector<1x!ii32>} : () -> ()
"test.op"() {"attr" = #attr} : () -> ()
"test.op"() {"attr" = [#attr]} : () -> ()

// CHECK:      "test.op"() {attr = 1 : i32} : () -> ()
// CHECK-NEXT: "test.op"() {attr = i32} : () -> ()
// CHECK-NEXT: "test.op"() {attr = vector<1xi32>} : () -> ()
// CHECK-NEXT: "test.op"() {attr = 0 : i32} : () -> ()
// CHECK-NEXT: "test.op"() {attr = [0 : i32]} : () -> ()


// -----

#attr = 0

"test.op"() {"attr" = #attr : i32} : () -> ()
// CHECK: '}' expected

// -----

"test.op"() {"attr" = #attr} : () -> ()
// CHECK: undefined symbol alias '#attr'
