// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsf64 = "test.op"() : () -> (f64)

%absf0 = math.absf %lhsf32 : f32
// CHECK:      {{%.*}} = math.absf {{%.*}}#0 : f32

%absi0 = math.absi %lhsi32: i32
// CHECK:      {{%.*}} = math.absi {{%.*}}#0 : i32
