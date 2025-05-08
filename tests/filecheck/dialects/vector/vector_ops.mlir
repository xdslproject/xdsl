// RUN: XDSL_ROUNDTRIP

func.func private @vector_test(%base : memref<4x4xindex>, %vec : vector<1xi1>, %i : index, %fvec : vector<2xf32>) {
  %load = vector.load %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
  vector.store %load, %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
  %load_nontemporal = vector.load %base[%i, %i] {"nontemporal" = false} : memref<4x4xindex>, vector<2xindex>
  vector.store %load_nontemporal, %base[%i, %i] {"nontemporal" = false} : memref<4x4xindex>, vector<2xindex>
  %load_nontemporal_1 = vector.load %base[%i, %i] {"nontemporal" = true} : memref<4x4xindex>, vector<2xindex>
  vector.store %load_nontemporal_1, %base[%i, %i] {"nontemporal" = true} : memref<4x4xindex>, vector<2xindex>
  %broadcast = vector.broadcast %i : index to vector<1xindex>
  %fma = vector.fma %fvec, %fvec, %fvec : vector<2xf32>
  %masked_load = vector.maskedload %base[%i, %i], %vec, %broadcast : memref<4x4xindex>, vector<1xi1>, vector<1xindex> into vector<1xindex>
  vector.maskedstore %base[%i, %i], %vec, %masked_load : memref<4x4xindex>, vector<1xi1>, vector<1xindex>
  "vector.print"(%masked_load) : (vector<1xindex>) -> ()
  %mask = vector.create_mask %i : vector<2xi1>
  %read = vector.transfer_read %base[%i, %i], %i {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<4x4xindex>, vector<4xindex>
  %read_1 = vector.transfer_read %base[%i, %i], %i {in_bounds = [true]} : memref<4x4xindex>, vector<4xindex>
  vector.transfer_write %read, %base[%i, %i] {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<4xindex>, memref<4x4xindex>
  func.return
}


// CHECK:       func.func private @vector_test(%base : memref<4x4xindex>, %vec : vector<1xi1>, %i : index, %fvec : vector<2xf32>) {
// CHECK-NEXT:     %load = vector.load %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     vector.store %load, %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     %load_nontemporal = vector.load %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     vector.store %load_nontemporal, %base[%i, %i] : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     %load_nontemporal_1 = vector.load %base[%i, %i] {nontemporal = true} : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     vector.store %load_nontemporal_1, %base[%i, %i] {nontemporal = true} : memref<4x4xindex>, vector<2xindex>
// CHECK-NEXT:     %broadcast = vector.broadcast %i : index to vector<1xindex>
// CHECK-NEXT:     %fma = vector.fma %fvec, %fvec, %fvec : vector<2xf32>
// CHECK-NEXT:     %masked_load = vector.maskedload %base[%i, %i], %vec, %broadcast : memref<4x4xindex>, vector<1xi1>, vector<1xindex> into vector<1xindex>
// CHECK-NEXT:     vector.maskedstore %base[%i, %i], %vec, %masked_load : memref<4x4xindex>, vector<1xi1>, vector<1xindex>
// CHECK-NEXT:     "vector.print"(%masked_load) : (vector<1xindex>) -> ()
// CHECK-NEXT:     %mask = vector.create_mask %i : vector<2xi1>
// CHECK-NEXT:     %read = vector.transfer_read %base[%i, %i], %i {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<4x4xindex>, vector<4xindex>
// CHECK-NEXT:     %read_1 = vector.transfer_read %base[%i, %i], %i {in_bounds = [true]} : memref<4x4xindex>, vector<4xindex>
// CHECK-NEXT:     vector.transfer_write %read, %base[%i, %i] {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<4xindex>, memref<4x4xindex>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }

// CHECK-LABEL: @extract_const_idx
func.func @extract_const_idx(%arg0: vector<4x8x16xf32>)
                             -> (vector<4x8x16xf32>, vector<8x16xf32>, vector<16xf32>, f32) {
  // CHECK: vector.extract {{.*}}[] : vector<4x8x16xf32> from vector<4x8x16xf32>
  %0 = vector.extract %arg0[] : vector<4x8x16xf32> from vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract {{.*}}[3] : vector<8x16xf32> from vector<4x8x16xf32>
  %1 = vector.extract %arg0[3] : vector<8x16xf32> from vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract {{.*}}[3, 3] : vector<16xf32> from vector<4x8x16xf32>
  %2 = vector.extract %arg0[3, 3] : vector<16xf32> from vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract {{.*}}[3, 3, 3] : f32 from vector<4x8x16xf32>
  %3 = vector.extract %arg0[3, 3, 3] : f32 from vector<4x8x16xf32>
  return %0, %1, %2, %3 : vector<4x8x16xf32>, vector<8x16xf32>, vector<16xf32>, f32
}

// CHECK-LABEL: @extract_val_idx
func.func @extract_val_idx(%arg0: vector<4x8x16xf32>, %idx: index)
                           -> (vector<8x16xf32>, vector<16xf32>, f32) {
  // CHECK: vector.extract %{{.*}}[%{{.*}}] : vector<8x16xf32> from vector<4x8x16xf32>
  %0 = vector.extract %arg0[%idx] : vector<8x16xf32> from vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[%{{.*}}, %{{.*}}] : vector<16xf32> from vector<4x8x16xf32>
  %1 = vector.extract %arg0[%idx, %idx] : vector<16xf32> from vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[%{{.*}}, 5, %{{.*}}] : f32 from vector<4x8x16xf32>
  %2 = vector.extract %arg0[%idx, 5, %idx] : f32 from vector<4x8x16xf32>
  return %0, %1, %2 : vector<8x16xf32>, vector<16xf32>, f32
}

// CHECK-LABEL: @extract_0d
func.func @extract_0d(%a: vector<f32>) -> f32 {
  // CHECK-NEXT: vector.extract %{{.*}}[] : f32 from vector<f32>
  %0 = vector.extract %a[] : f32 from vector<f32>
  return %0 : f32
}

// CHECK-LABEL: @extract_poison_idx
func.func @extract_poison_idx(%a: vector<4x5xf32>) -> f32 {
  // CHECK-NEXT: vector.extract %{{.*}}[-1, 0] : f32 from vector<4x5xf32>
  %0 = vector.extract %a[-1, 0] : f32 from vector<4x5xf32>
  return %0 : f32
}

// CHECK-LABEL: @insert_const_idx
func.func @insert_const_idx(%a: f32, %b: vector<16xf32>, %c: vector<8x16xf32>,
                            %res: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  // CHECK: vector.insert %{{.*}}, %{{.*}}[3] : vector<8x16xf32> into vector<4x8x16xf32>
  %1 = vector.insert %c, %res[3] : vector<8x16xf32> into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[3, 3] : vector<16xf32> into vector<4x8x16xf32>
  %2 = vector.insert %b, %res[3, 3] : vector<16xf32> into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[3, 3, 3] : f32 into vector<4x8x16xf32>
  %3 = vector.insert %a, %res[3, 3, 3] : f32 into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[] : vector<4x8x16xf32> into vector<4x8x16xf32>
  %4 = vector.insert %3, %3[] : vector<4x8x16xf32> into vector<4x8x16xf32>
  return %4 : vector<4x8x16xf32>
}

// CHECK-LABEL: @insert_val_idx
func.func @insert_val_idx(%a: f32, %b: vector<16xf32>, %c: vector<8x16xf32>,
                          %idx: index, %res: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  // CHECK: vector.insert %{{.*}}, %{{.*}}[%{{.*}}] : vector<8x16xf32> into vector<4x8x16xf32>
  %0 = vector.insert %c, %res[%idx] : vector<8x16xf32> into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<16xf32> into vector<4x8x16xf32>
  %1 = vector.insert %b, %res[%idx, %idx] : vector<16xf32> into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[%{{.*}}, 5, %{{.*}}] : f32 into vector<4x8x16xf32>
  %2 = vector.insert %a, %res[%idx, 5, %idx] : f32 into vector<4x8x16xf32>
  return %2 : vector<4x8x16xf32>
}

// CHECK-LABEL: @insert_0d
func.func @insert_0d(%a: f32, %b: vector<f32>, %c: vector<2x3xf32>) -> (vector<f32>, vector<2x3xf32>) {
  // CHECK-NEXT: vector.insert %{{.*}}, %{{.*}}[] : f32 into vector<f32>
  %1 = vector.insert %a,  %b[] : f32 into vector<f32>
  // CHECK-NEXT: vector.insert %{{.*}}, %{{.*}}[0, 1] : vector<f32> into vector<2x3xf32>
  %2 = vector.insert %b,  %c[0, 1] : vector<f32> into vector<2x3xf32>
  return %1, %2 : vector<f32>, vector<2x3xf32>
}

// CHECK-LABEL: @insert_poison_idx
func.func @insert_poison_idx(%a: vector<4x5xf32>, %b: f32) {
  // CHECK-NEXT: vector.insert %{{.*}}, %{{.*}}[-1, 0] : f32 into vector<4x5xf32>
  vector.insert %b, %a[-1, 0] : f32 into vector<4x5xf32>
  return
}
