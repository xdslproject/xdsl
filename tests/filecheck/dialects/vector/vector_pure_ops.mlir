// RUN: xdsl-opt %s -p cse | filecheck %s

%m0, %i0, %vf = "test.op"() : () -> (memref<4x4xindex>, index, vector<2xf32>)
%load = vector.load %m0[%i0, %i0] : memref<4x4xindex>, vector<2xindex>
vector.store %load, %m0[%i0, %i0] : memref<4x4xindex>, vector<2xindex>
%broadcast = vector.broadcast %i0 : index to vector<1xindex>
%fma = vector.fma %vf, %vf, %vf : vector<2xf32>
%extract_op = "vector.extractelement"(%broadcast, %i0) : (vector<1xindex>, index) -> index
"vector.insertelement"(%extract_op, %broadcast, %i0) : (index, vector<1xindex>, index) -> vector<1xindex>
/// Check that unused results from vector.broadcast and vector.fma are eliminated
// CHECK:       %m0, %i0, %vf = "test.op"() : () -> (memref<4x4xindex>, index, vector<2xf32>)
// CHECK-NEXT:  %load = vector.load %m0[%i0, %i0] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:  vector.store %load, %m0[%i0, %i0] : memref<4x4xindex>, vector<2xindex>
