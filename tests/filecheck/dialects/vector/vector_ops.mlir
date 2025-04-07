// RUN: XDSL_ROUNDTRIP

func.func private @vector_test(%base : memref<4x4xindex>, %vec : vector<1xi1>, %i : index, %fvec : vector<2xf32>) {
  %load = vector.load %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
  vector.store %load, %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
  %broadcast = vector.broadcast %i : index to vector<1xindex>
  %fma = vector.fma %fvec, %fvec, %fvec : vector<2xf32>
  %masked_load = vector.maskedload %base[%i, %i], %vec, %broadcast : memref<4x4xindex>, vector<1xi1>, vector<1xindex> into vector<1xindex>
  vector.maskedstore %base[%i, %i], %vec, %masked_load : memref<4x4xindex>, vector<1xi1>, vector<1xindex>
  "vector.print"(%masked_load) : (vector<1xindex>) -> ()
  %mask = vector.create_mask %i : vector<2xi1>
  func.return
}


// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @vector_test(%base : memref<4x4xindex>, %vec : vector<1xi1>, %i : index, %fvec : vector<2xf32>) {
// CHECK-NEXT:     %load = vector.load %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     vector.store %load, %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     %broadcast = vector.broadcast %i : index to vector<1xindex>
// CHECK-NEXT:     %fma = vector.fma %fvec, %fvec, %fvec : vector<2xf32>
// CHECK-NEXT:     %masked_load = vector.maskedload %base[%i, %i], %vec, %broadcast : memref<4x4xindex>, vector<1xi1>, vector<1xindex> into vector<1xindex>
// CHECK-NEXT:     vector.maskedstore %base[%i, %i], %vec, %masked_load : memref<4x4xindex>, vector<1xi1>, vector<1xindex>
// CHECK-NEXT:     "vector.print"(%masked_load) : (vector<1xindex>) -> ()
// CHECK-NEXT:     %mask = vector.create_mask %i : vector<2xi1>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
