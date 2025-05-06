// RUN: xdsl-opt --verify-diagnostics %s | filecheck %s

%0 = linalg.index 3 : index

// CHECK: Operation does not verify: 'linalg.index' expects parent op 'linalg.generic'
