// RUN: xdsl-opt --parsing-diagnostics --split-input-file %s | filecheck %s --strict-whitespace --match-full-lines

"test.op"(abc) : () -> ()
//      CHECK:{{\S*}}:3:11
// CHECK-NEXT:"test.op"(abc) : () -> ()
// CHECK-NEXT:          ^^^
// CHECK-NEXT:          operand expected

// -----

test.op : () -> ()

//      CHECK:{{\S*}}:11:9
// CHECK-NEXT:test.op : () -> ()
// CHECK-NEXT:        ^
// CHECK-NEXT:        Operation test.op does not have a custom format.

// -----

module {
    "test.op"() [^unknown_successor]: () -> ()
}

//      CHECK:    "test.op"() [^unknown_successor]: () -> ()
// CHECK-NEXT:                 ^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:                 reference to block "^unknown_successor" without implementation

// -----

//      CHECK:"test.op"() : () -> !emitc.ptr<"bad">
// CHECK-NEXT:                              ^^^^^^^
// CHECK-NEXT:                              parameter 'pointee_type' does not verify:
// CHECK-NEXT:"bad" should be of attribute subclassing `TypeAttribute`
"test.op"() : () -> !emitc.ptr<"bad">

// -----

//      CHECK:"test.op"() : () -> !emitc.ptr<!emitc.lvalue<i32>>
// CHECK-NEXT:                              ^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:                              pointers to lvalues are not allowed
"test.op"() : () -> !emitc.ptr<!emitc.lvalue<i32>>
