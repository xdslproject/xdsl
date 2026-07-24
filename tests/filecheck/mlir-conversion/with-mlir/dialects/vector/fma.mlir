// RUN: xdsl-opt --print-op-generic %s | $XDSL_MLIR_OPT --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | $XDSL_MLIR_OPT --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s


%vf = "test.op"() : () -> vector<2xf32>
%fma = vector.fma %vf, %vf, %vf : vector<2xf32>

// CHECK: %1 = vector.fma %0, %0, %0 : vector<2xf32>
