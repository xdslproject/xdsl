// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

%vector0, %vector1, %i0 = "test.op"() : () -> (vector<index>, vector<3xindex>, index)
// CHECK:      %0, %1, %2 = "test.op"() : () -> (vector<index>, vector<3xindex>, index)

%0 = "vector.insertelement"(%i0, %vector0) : (index, vector<index>) -> vector<index>
// CHECK-NEXT: %3 = "vector.insertelement"(%2, %0) : (index, vector<index>) -> vector<index>

%1 = "vector.insertelement"(%i0, %vector1, %i0) : (index, vector<3xindex>, index) -> vector<3xindex>
// CHECK-NEXT: %4 = "vector.insertelement"(%2, %1, %2) : (index, vector<3xindex>, index) -> vector<3xindex>

%2 = "vector.extractelement"(%vector1, %i0) : (vector<3xindex>, index) -> index
// CHECK-NEXT: %5 = "vector.extractelement"(%1, %2) : (vector<3xindex>, index) -> index

%3 = "vector.extractelement"(%vector0) : (vector<index>) -> index
// CHECK-NEXT: %6 = "vector.extractelement"(%0) : (vector<index>) -> index
