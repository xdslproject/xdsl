// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s

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
