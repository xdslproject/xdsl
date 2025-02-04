// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

"test.op"() {key1, key1} : () -> ()
// CHECK: Duplicate key 'key1' in dictionary attribute

// -----

"test.op"() <{key1, key1}> : () -> ()
// CHECK: Duplicate key 'key1' in properties dictionary 
