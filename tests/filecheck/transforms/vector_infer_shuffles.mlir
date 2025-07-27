// RUN: xdsl-opt -p vector-infer-shuffles --split-input-file %s | filecheck %s

// CHECK: "test.op"() : () -> vector<4xf32>
%v = "test.op"() : () -> vector<4xf32>
// CHECK-NEXT: %f = vector.extract %v[1] : f32 from vector<4xf32>
%f = vector.extract %v[1] : f32 from vector<4xf32>
// CHECK-NEXT: %r = vector.shuffle %v, %v [1, 1, 1, 1, 1] : vector<4xf32>, vector<4xf32>
%r = vector.broadcast %f : f32 to vector<5xf32>
// CHECK-NEXT: "test.op"(%r) : (vector<5xf32>) -> ()
"test.op"(%r) : (vector<5xf32>) -> ()

// %v1, %v2, %i = "test.op"() : () -> (vector<4xf32>, vector<2x2xf32>, index)
// %extracted_vector = vector.extract %v2 [0] : vector<2xf32> from vector<2x2xf32>
// %broadcast_vector = vector.broadcast %extracted_vector : vector<2xf32> to vector<2x2xf32>
