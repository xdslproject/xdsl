// RUN: xdsl-opt %s --verify-diagnostics -p cse | filecheck %s

%m0, %i0 = "test.op"() : () -> (memref<4x4xindex>, index)
%load = "vector.load"(%m0, %i0, %i0) : (memref<4x4xindex>, index, index) -> vector<2xindex>
"vector.store"(%load, %m0, %i0, %i0) : (vector<2xindex>, memref<4x4xindex>, index, index) -> ()
%broadcast = "vector.broadcast"(%i0) : (index) -> vector<1xindex>
%fma = "vector.fma"(%load, %load, %load) : (vector<2xindex>, vector<2xindex>, vector<2xindex>) -> vector<2xindex>

/// Check that unused results from vector.broadcast and vector.fma are eliminated
// CHECK:       %m0, %i0 = "test.op"() : () -> (memref<4x4xindex>, index) 
// CHECK-NEXT:  %load = "vector.load"(%m0, %i0, %i0) : (memref<4x4xindex>, index, index) -> vector<2xindex>
// CHECK-NEXT:  "vector.store"(%load, %m0, %i0, %i0) : (vector<2xindex>, memref<4x4xindex>, index, index) -> ()
