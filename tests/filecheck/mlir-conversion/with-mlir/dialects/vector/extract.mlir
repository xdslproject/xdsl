// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s

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
