// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

// Each struct parameter of omp.declaretarget may appear at most once.
"test.op"() {x = #omp.declaretarget<automap = true, automap = false>} : () -> ()
// CHECK: duplicate struct parameter name: automap

// -----

"test.op"() {x = #omp.declaretarget<device_type = (any), device_type = (host)>} : () -> ()
// CHECK: duplicate struct parameter name: device_type

// -----

// Unknown struct parameters are rejected.
"test.op"() {x = #omp.declaretarget<unknown = true>} : () -> ()
// CHECK: Expected one of 'automap', 'capture_clause', 'device_type'

// -----

// automap is a boolean.
"test.op"() {x = #omp.declaretarget<automap = 1>} : () -> ()
// CHECK: Expected boolean literal
