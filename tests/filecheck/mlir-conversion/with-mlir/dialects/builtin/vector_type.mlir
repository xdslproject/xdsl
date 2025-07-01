// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

// CHECK:      vector<1xindex>
"test.op"() {v.v1 = vector<1xindex>} : () -> ()

// CHECK:      vector<[2]xindex>
"test.op"() {v.vs2 = vector<[2]xindex>} : () -> ()

// CHECK:      vector<[3]x4xindex>
"test.op"() {v.vs34 = vector<[3]x4xindex>} : () -> ()

// CHECK:      vector<[5]x[6]xindex>
"test.op"() {v.vs5s6 = vector<[5]x[6]xindex>} : () -> ()

// CHECK:      vector<7x[8]xindex>
"test.op"() {v.v7s8 = vector<7x[8]xindex>} : () -> ()
