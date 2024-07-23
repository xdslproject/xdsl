// RUN: xdsl-opt %s -p memref-stream-legalize | filecheck %s

func.func public @sumvf64(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
  memref_stream.generic {
    bounds = [8, 16],
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
  ^0(%in : vector<1xf64>, %in_1 : vector<1xf64>, %out : vector<1xf64>):
    %0 = arith.addf %in, %in_1 : vector<1xf64>
    memref_stream.yield %0 : vector<1xf64>
  }
  func.return %arg2 : memref<8x16xf64>
}

// CHECK:       func.func public @sumvf64(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [8, 16],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:  } ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
// CHECK-NEXT:  ^0(%in : vector<1xf64>, %in_1 : vector<1xf64>, %out : vector<1xf64>):
// CHECK-NEXT:    %0 = arith.addf %in, %in_1 : vector<1xf64>
// CHECK-NEXT:    memref_stream.yield %0 : vector<1xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.return %arg2 : memref<8x16xf64>
// CHECK-NEXT:}


func.func public @sumf32(%arg0 : memref<8x16xf32>, %arg1 : memref<8x16xf32>, %arg2 : memref<8x16xf32>) -> memref<8x16xf32> {
  memref_stream.generic {
    bounds = [8, 16],
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : memref<8x16xf32>, memref<8x16xf32>) outs(%arg2 : memref<8x16xf32>) {
  ^0(%in : f32, %in_1 : f32, %out : f32):
    %0 = arith.addf %in, %in_1 : f32
    memref_stream.yield %0 : f32
  }
  func.return %arg2 : memref<8x16xf32>
}

// CHECK:         func.func public @sumf32(%arg0 : memref<8x16xf32>, %arg1 : memref<8x16xf32>, %arg2 : memref<8x16xf32>) -> memref<8x16xf32> {
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [8, 8],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:    } ins(%arg0, %arg1 : memref<8x16xf32>, memref<8x16xf32>) outs(%arg2 : memref<8x16xf32>) {
// CHECK-NEXT:    ^0(%0 : vector<2xf32>, %1 : vector<2xf32>, %2 : vector<2xf32>):
// CHECK-NEXT:      %3 = arith.addf %0, %1 : vector<2xf32>
// CHECK-NEXT:      memref_stream.yield %3 : vector<2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %arg2 : memref<8x16xf32>
// CHECK-NEXT:  }


func.func public @sumf16(%arg0 : memref<8x16xf16>, %arg1 : memref<8x16xf16>, %arg2 : memref<8x16xf16>) -> memref<8x16xf16> {
  memref_stream.generic {
    bounds = [8, 16],
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>) outs(%arg2 : memref<8x16xf16>) {
  ^0(%in : f16, %in_1 : f16, %out : f16):
    %0 = arith.addf %in, %in_1 : f16
    memref_stream.yield %0 : f16
  }
  func.return %arg2 : memref<8x16xf16>
}

// CHECK:         func.func public @sumf16(%arg0 : memref<8x16xf16>, %arg1 : memref<8x16xf16>, %arg2 : memref<8x16xf16>) -> memref<8x16xf16> {
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [8, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:        affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:    } ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>) outs(%arg2 : memref<8x16xf16>) {
// CHECK-NEXT:    ^0(%0 : vector<4xf16>, %1 : vector<4xf16>, %2 : vector<4xf16>):
// CHECK-NEXT:      %3 = arith.addf %0, %1 : vector<4xf16>
// CHECK-NEXT:      memref_stream.yield %3 : vector<4xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %arg2 : memref<8x16xf16>
// CHECK-NEXT:  }
// CHECK-NEXT:}

func.func public @chainf16(%arg0 : memref<8x16xf16>, %arg1 : memref<8x16xf16>, %arg2 : memref<8x16xf16>) -> memref<8x16xf16> {
  memref_stream.generic {
    bounds = [8, 16],
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>) outs(%arg2 : memref<8x16xf16>) {
  ^0(%in : f16, %in_1 : f16, %out : f16):
    %0 = arith.addf %in, %in_1 : f16
    %1 = arith.mulf %0, %in_1 : f16
    %2 = arith.divf %0, %1 : f16
    %3 = arith.mulf %2, %1 : f16
    %4 = arith.divf %3, %2 : f16
    %5 = arith.mulf %4, %3 : f16
    %6 = arith.divf %4, %5 : f16
    %7 = arith.mulf %5, %6 : f16
    memref_stream.yield %7 : f16
  }
  func.return %arg2 : memref<8x16xf16>
}
