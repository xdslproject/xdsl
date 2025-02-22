// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

// CHECK:      vector<3xindex>
builtin.module attributes {v.v3 = vector<3xindex>} {}

builtin.module attributes {v.vs3 = vector<[3]xindex>} {}

builtin.module attributes {v.vs34 = vector<[3]x4xindex>} {}

builtin.module attributes {v.vs3s4 = vector<[3]x[4]xindex>} {}

builtin.module attributes {v.v3s4 = vector<3x[4]xindex>} {}
