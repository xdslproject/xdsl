// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s


%vf = "test.op"() : () -> vector<2xf32>
%cast = vector.bitcast %vf : vector<2xf32> to vector<2xi32>

// CHECK: %1 = vector.bitcast %0 : vector<2xf32> to vector<2xi32>
