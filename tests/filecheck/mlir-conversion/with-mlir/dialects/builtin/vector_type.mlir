// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

// CHECK:      vector<1xindex>
builtin.module attributes {v.v1 = vector<1xindex>} {}

// CHECK:      vector<[2]xindex>
builtin.module attributes {v.vs2 = vector<[2]xindex>} {}

// CHECK:      vector<[3]x4xindex>
builtin.module attributes {v.vs34 = vector<[3]x4xindex>} {}

// CHECK:      vector<[5]x[6]xindex>
builtin.module attributes {v.vs5s6 = vector<[5]x[6]xindex>} {}

// CHECK:      vector<7x[8]xindex>
builtin.module attributes {v.v7s8 = vector<7x[8]xindex>} {}
