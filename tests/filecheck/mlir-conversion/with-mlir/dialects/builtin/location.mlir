// RUN: xdsl-opt --print-op-generic %s --print-debuginfo | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect --mlir-print-debuginfo --mlir-print-local-scope | xdsl-opt --print-op-generic --print-debuginfo | filecheck %s

"test.op"() : () -> () loc(unknown)

// CHECK: "test.op"() : () -> () loc(unknown)
