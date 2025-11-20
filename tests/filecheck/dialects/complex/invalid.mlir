// RUN: xdsl-opt --split-input-file %s --verify-diagnostics | filecheck %s

%i64 = "test.op"() : () -> i64
%0 = complex.bitcast %i64 : i64 to f64
//CHECK: Expected ('i64', 'f64') to be bitcast between complex and equal arith types

// -----

%f64 = "test.op"() : () -> f64
%0 = complex.bitcast %f64 : f64 to i32
//CHECK: Expected ('f64', 'i32') to be bitcast between complex and equal arith types

// -----

%f64 = "test.op"() : () -> f64
%0 = complex.bitcast %f64 : f64 to f32
//CHECK: Expected ('f64', 'f32') to be bitcast between complex and equal arith types

// -----

%complex = "test.op"() : () -> (complex<f32>)
%0 = complex.bitcast %complex : complex<f32> to i32
//CHECK: Expected ('complex<f32>', 'i32') to be bitcast between complex and equal arith types
