// Open Earth Compiler's laplace implementation.

// Following Affine maps are used for accessing the hyper-rectangular region created by memref 
// subviews in oec implementation.
// Consider memref dimensions as [m_dim1, m_dim2, m_dim3] and subview offsets, subview dimensions 
// as [s_offset1, s_offset2, s_offset3], [s_dim1, s_dim2, s_dim3] respectibely.
// Then corresponding affine map annotation to access (i, j, k) element in subview will be:
// i x m_dim2 x m_dim3 + j x m_dim3 + k + s_offset1 x m_dim2 x m_dim3 + 
// s_offset2 x m_dim3 + s_offset3.
#map0 = affine_map<(d0, d1, d2) -> (d0 * 5184 + d1 * 72 + d2 + 20955)>
#map1 = affine_map<(d0, d1, d2) -> (d0 * 5184 + d1 * 72 + d2 + 21028)>

func.func @laplace_oec(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant -4.000000e+00 : f64
    %0 = memref.cast %arg0 : memref<?x?x?xf64> to memref<72x72x72xf64>
    %1 = memref.cast %arg1 : memref<?x?x?xf64> to memref<72x72x72xf64>
    %2 = memref.subview %0[4, 3, 3] [64, 66, 66] [1, 1, 1] : memref<72x72x72xf64> to memref<64x66x66xf64, #map0>
    %3 = memref.subview %1[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, #map1>
    scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c64, %c64, %c64) step (%c1, %c1, %c1) {
      %4 = arith.addi %arg3, %c1 : index
      %5 = memref.load %2[%arg4, %4, %arg2] : memref<64x66x66xf64, #map0>
      %6 = arith.addi %arg2, %c2 : index
      %7 = arith.addi %arg3, %c1 : index
      %8 = memref.load %2[%arg4, %7, %6] : memref<64x66x66xf64, #map0>
      %9 = arith.addi %arg2, %c1 : index
      %10 = arith.addi %arg3, %c2 : index
      %11 = memref.load %2[%arg4, %10, %9] : memref<64x66x66xf64, #map0>
      %12 = arith.addi %arg2, %c1 : index
      %13 = memref.load %2[%arg4, %arg3, %12] : memref<64x66x66xf64, #map0>
      %14 = arith.addi %arg2, %c1 : index
      %15 = arith.addi %arg3, %c1 : index
      %16 = memref.load %2[%arg4, %15, %14] : memref<64x66x66xf64, #map0>
      %17 = arith.addf %5, %8 : f64
      %18 = arith.addf %11, %13 : f64
      %19 = arith.addf %17, %18 : f64
      %20 = arith.mulf %16, %cst : f64
      %21 = arith.addf %20, %19 : f64
      memref.store %21, %3[%arg4, %arg3, %arg2] : memref<64x64x64xf64, #map1>
      scf.yield
    }
    return
  }
