// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

func.func @insert_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // CHECK: Expected position attribute rank (6) to match dest vector rank (3).
  %1 = vector.insert %a, %b[3, 3, 3, 3, 3, 3] : f32 into vector<4x8x16xf32>
  func.return
}

// -----

func.func @insert_vector_type(%a: vector<4xf32>, %b: vector<4x8x16xf32>) {
  // CHECK: Expected position attribute rank (1) + source rank (1) to match dest vector rank (3).
  %1 = vector.insert %a, %b[3] : vector<4xf32> into vector<4x8x16xf32>
  func.return
}

// -----

func.func @insert_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // CHECK: Expected position attribute rank (2) to match dest vector rank (3).
  %1 = vector.insert %a, %b[3, 3] : f32 into vector<4x8x16xf32>
  func.return
}

// -----

func.func @insert_0d(%a: vector<f32>, %b: vector<4x8x16xf32>) {
  // CHECK: Cannot insert 0d vector.
  %1 = vector.insert %a, %b[2, 6] : vector<f32> into vector<4x8x16xf32>
  func.return
}

// -----

func.func @insert_0d(%a: f32, %b: vector<f32>) {
  // CHECK: Expected position attribute rank (1) to match dest vector rank (0).
  %1 = vector.insert %a, %b[0] : f32 into vector<f32>
  func.return
}

// -----

func.func @extract_position_rank_overflow(%arg0: vector<4x8x16xf32>) {
  // CHECK: Expected position attribute rank (4) to match source vector rank (3).
  %1 = vector.extract %arg0[0, 0, 0, 0] : f32 from vector<4x8x16xf32>
  func.return
}

// -----

func.func @extract_position_rank_overflow_generic(%arg0: vector<4x8x16xf32>) {
  // CHECK: Expected position attribute rank (4) + result rank (1) to match source vector rank (3).
  %1 = "vector.extract" (%arg0) <{static_position = array<i64: 0, 0, 0, 0>}> : (vector<4x8x16xf32>) -> (vector<16xf32>)
  func.return
}

// -----

func.func @extract_0d(%arg0: vector<f32>) {
  // CHECK: Expected position attribute rank (1) to match source vector rank (0).
  %1 = vector.extract %arg0[0] : f32 from vector<f32>
  func.return
}

// -----

func.func @extract_0d(%arg0: vector<f32>) {
  // CHECK: Cannot extract 0d vector.
  %1 = vector.extract %arg0[0] : vector<f32> from vector<4xf32>
  func.return
}
