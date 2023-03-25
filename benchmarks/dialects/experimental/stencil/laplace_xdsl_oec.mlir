// For CPU lowering, use the following command:
// mlir-opt %s --lower-affine --arith-expand --convert-scf-to-cf --expand-strided-metadata --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=path-to-libmlir_c_runner_utils.so

// Affine maps used by oec's implementation.
#map0 = affine_map<(d0, d1, d2) -> (d0 * 5184 + d1 * 72 + d2 + 20955)>
#map1 = affine_map<(d0, d1, d2) -> (d0 * 5184 + d1 * 72 + d2 + 21028)>

builtin.module {
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @rtclock() -> f64

  // Contains oec's laplace lowered IR.
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

  // Contains xdsl's laplace lowered IR.
  func.func @laplace_xdsl(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) attributes {stencil.program} {
    %0 = "arith.constant"() {value = -4.000000e+00 : f64} : () -> f64
    %1 = "arith.constant"() {value = 5 : index} : () -> index
    %2 = "arith.constant"() {value = 3 : index} : () -> index
    %3 = "arith.constant"() {value = 4 : index} : () -> index
    %4 = "arith.constant"() {value = 64 : index} : () -> index
    %5 = "arith.constant"() {value = 1 : index} : () -> index
    %6 = "arith.constant"() {value = 0 : index} : () -> index
    "scf.parallel"(%6, %6, %6, %4, %4, %4, %5, %5, %5) ({
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      %7 = "arith.addi"(%arg4, %3) : (index, index) -> index
      %8 = "arith.addi"(%arg3, %3) : (index, index) -> index
      %9 = "arith.addi"(%arg2, %2) : (index, index) -> index
      %10 = "memref.load"(%arg0, %7, %8, %9) : (memref<?x?x?xf64>, index, index, index) -> f64
      %11 = "arith.addi"(%arg4, %3) : (index, index) -> index
      %12 = "arith.addi"(%arg3, %3) : (index, index) -> index
      %13 = "arith.addi"(%arg2, %1) : (index, index) -> index
      %14 = "memref.load"(%arg0, %11, %12, %13) : (memref<?x?x?xf64>, index, index, index) -> f64
      %15 = "arith.addi"(%arg4, %3) : (index, index) -> index
      %16 = "arith.addi"(%arg3, %1) : (index, index) -> index
      %17 = "arith.addi"(%arg2, %3) : (index, index) -> index
      %18 = "memref.load"(%arg0, %15, %16, %17) : (memref<?x?x?xf64>, index, index, index) -> f64
      %19 = "arith.addi"(%arg4, %3) : (index, index) -> index
      %20 = "arith.addi"(%arg3, %2) : (index, index) -> index
      %21 = "arith.addi"(%arg2, %3) : (index, index) -> index
      %22 = "memref.load"(%arg0, %19, %20, %21) : (memref<?x?x?xf64>, index, index, index) -> f64
      %23 = "arith.addi"(%arg4, %3) : (index, index) -> index
      %24 = "arith.addi"(%arg3, %3) : (index, index) -> index
      %25 = "arith.addi"(%arg2, %3) : (index, index) -> index
      %26 = "memref.load"(%arg0, %23, %24, %25) : (memref<?x?x?xf64>, index, index, index) -> f64
      %27 = "arith.addf"(%10, %14) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      %28 = "arith.addf"(%18, %22) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      %29 = "arith.addf"(%27, %28) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      %30 = "arith.mulf"(%26, %0) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      %31 = "arith.addf"(%30, %29) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      %32 = "arith.addi"(%arg4, %3) : (index, index) -> index
      %33 = "arith.addi"(%arg3, %3) : (index, index) -> index
      %34 = "arith.addi"(%arg2, %3) : (index, index) -> index
      "memref.store"(%31, %arg1, %32, %33, %34) : (f64, memref<?x?x?xf64>, index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
    return
  }

  // Funtion for filling a 3d f64 memref with a constant.
  func.func @alloc_3d_filled_f64(%arg0: index, %arg1: index, %arg2: index, %arg3: f64) -> memref<?x?x?xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2) : memref<?x?x?xf64>
    scf.for %arg4 = %c0 to %arg0 step %c1 {
      scf.for %arg5 = %c0 to %arg1 step %c1 {
        scf.for %arg6 = %c0 to %arg2 step %c1 {
          memref.store %arg3, %0[%arg4, %arg5, %arg6] : memref<?x?x?xf64>
        }
      }
    }
    return %0 : memref<?x?x?xf64>
  }

  // Function for comparing 2 3d f64 memrefs.
  func.func @compare_3d_memref_f64(%dim1: index, %dim2: index, %dim3: index, %memref1 : memref<?x?x?xf64>, %memref2 : memref<?x?x?xf64>) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %dim1 step %c1 {
      scf.for %j = %c0 to %dim2 step %c1 {
        scf.for %k = %c0 to %dim3 step %c1 {
          %val1 = memref.load %memref1[%i, %j, %k] : memref<?x?x?xf64>
          %val2 = memref.load %memref2[%i, %j, %k] : memref<?x?x?xf64>

          %check_val = arith.cmpf one, %val1, %val2 : f64
          scf.if %check_val {
            // When different values are found, print their difference.
            %check_diff = arith.subf %val1, %val2 : f64
            vector.print %check_diff : f64
          } 
        }
      }
    }

    // Return 1 after successful execution.
    %true_val = arith.constant 1 : i1
    return %true_val : i1
  }

  func.func @main() {
    %memref1_size1 = "arith.constant"() {"value" = 72 : index} : () -> index
    %memref1_size2 = "arith.constant"() {"value" = 72 : index} : () -> index
    %memref1_size3 = "arith.constant"() {"value" = 72 : index} : () -> index
    %memref1_elem = "arith.constant"() {"value" = 7.0 : f64} : () -> f64
    %memref2_elem = "arith.constant"() {"value" = 9.0 : f64} : () -> f64

    // Get input memref for both implementations.
    %memref1 = func.call @alloc_3d_filled_f64(%memref1_size1, %memref1_size2, %memref1_size3, %memref1_elem) : (index, index, index, f64) -> memref<?x?x?xf64>

    // Get output memref for xdsl implementation.
    %memref2_xdsl = func.call @alloc_3d_filled_f64(%memref1_size1, %memref1_size2, %memref1_size3, %memref2_elem) : (index, index, index, f64) -> memref<?x?x?xf64>

    // Get output memref for oec implementation.
    %memref2_oec = func.call @alloc_3d_filled_f64(%memref1_size1, %memref1_size2, %memref1_size3, %memref2_elem) : (index, index, index, f64) -> memref<?x?x?xf64>

    // Execution count.
    %reps = arith.constant 100 : index

    // Record start time.
    %t_start = func.call @rtclock() : () -> f64

    // Execute laplace_xdsl for "reps" times.
    affine.for %arg0 = 0 to %reps {
      func.call @laplace_xdsl(%memref1, %memref2_xdsl) : (memref<?x?x?xf64>, memref<?x?x?xf64>) -> ()
    }

    // Record end time for laplace_xdsl.
    %t_end_laplace_xdsl = func.call @rtclock() : () -> f64

    // Execute laplace_oec for "reps" times.
    affine.for %arg0 = 0 to %reps {
      func.call @laplace_oec(%memref1, %memref2_oec) : (memref<?x?x?xf64>, memref<?x?x?xf64>) -> ()
    }

    // Record end time for laplace_oec.
    %t_end_laplace_oec = func.call @rtclock() : () -> f64

    // Get the total running time for laplace_xdsl.
    %t_laplace_xdsl = arith.subf %t_end_laplace_xdsl, %t_start : f64

    // Get the total running time for laplace_oec.
    %t_laplace_oec = arith.subf %t_end_laplace_oec, %t_end_laplace_xdsl : f64

    // Print total time taken by laplace_xdsl.
    vector.print %t_laplace_xdsl : f64

    // Print total time taken by laplace_oec.
    vector.print %t_laplace_oec : f64

    // Optionally print input and output memrefs.
    // %print_memref1 = memref.cast %memref1 : memref<?x?x?xf64> to memref<*xf64>
    // func.call @printMemrefF64(%print_memref1) : (memref<*xf64>) -> ()

    // %print_memref2_xdsl = memref.cast %memref2_xdsl : memref<?x?x?xf64> to memref<*xf64>
    // func.call @printMemrefF64(%print_memref2_xdsl) : (memref<*xf64>) -> ()

    // %print_memref2_oec = memref.cast %memref2_oec : memref<?x?x?xf64> to memref<*xf64>
    // func.call @printMemrefF64(%print_memref2_oec) : (memref<*xf64>) -> ()

    // Compare output produced by both implementations.
    %check_eq_memrefs = func.call @compare_3d_memref_f64(%memref1_size1, %memref1_size2, %memref1_size3, %memref2_xdsl, %memref2_oec) : (index, index, index, memref<?x?x?xf64>, memref<?x?x?xf64>) -> i1
    vector.print %check_eq_memrefs : i1

    func.return
  }
}
