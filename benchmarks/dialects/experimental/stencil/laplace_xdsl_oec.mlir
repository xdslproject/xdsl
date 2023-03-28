// File for benchmarking and comparing xDSL and Open Earth Compiler's laplace implementations.

builtin.module {
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @rtclock() -> f64

  // Function declaration for oec's laplace lowered IR.
  func.func private @laplace_oec(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) 

  // Function declaration for xdsl's laplace lowered IR.
  func.func private @laplace_xdsl(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>) attributes {stencil.program} 

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
