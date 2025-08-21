// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s

%vf, %acc = "test.op"() : () -> (vector<2xf32>, f32)

%sum = vector.reduction <add>, %vf : vector<2xf32> into f32
%sum_1 = vector.reduction <add>, %vf, %acc : vector<2xf32> into f32

// CHECK: %2 = vector.reduction <add>, %0 : vector<2xf32> into f32
// CHECK: %3 = vector.reduction <add>, %0, %1 : vector<2xf32> into f32
