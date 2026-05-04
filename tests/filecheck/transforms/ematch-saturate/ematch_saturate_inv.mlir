// first layer
func.func @get_inv(%A: memref<4x4xf32>) -> memref<4x4xf32> {
  // Pass the input to the inverse function, expecting the new SSA value back
  %inv = func.call @matrix_inverse_4x4(%A) : (memref<4x4xf32>) -> memref<4x4xf32>
  return %inv : memref<4x4xf32>
}

func.func @matrix_inverse_4x4(%input: memref<4x4xf32>) -> memref<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f1 = arith.constant 1.0 : f32
  %f0 = arith.constant 0.0 : f32

  // Allocate the new output matrix
  %output = memref.alloc() : memref<4x4xf32>

  // Allocate a temporary working matrix so we don't destroy the original %input
  %working_input = memref.alloc() : memref<4x4xf32>

  // 1. Initialize output as Identity and copy %input into %working_input
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      // Identity init
      %is_diag = arith.cmpi eq, %i, %j : index
      %val = arith.select %is_diag, %f1, %f0 : f32
      memref.store %val, %output[%i, %j] : memref<4x4xf32>

      // Copy input to working buffer
      %in_val = memref.load %input[%i, %j] : memref<4x4xf32>
      memref.store %in_val, %working_input[%i, %j] : memref<4x4xf32>
    }
  }

  // 2. Gauss-Jordan Elimination (operating on %working_input and %output)
  scf.for %k = %c0 to %c4 step %c1 {
    // --- Step A: Scale pivot row to 1.0 ---
    %pivot_val = memref.load %working_input[%k, %k] : memref<4x4xf32>

    scf.for %j = %c0 to %c4 step %c1 {
      // Scale Input Row
      %in_val = memref.load %working_input[%k, %j] : memref<4x4xf32>
      %scaled_in = arith.divf %in_val, %pivot_val : f32
      memref.store %scaled_in, %working_input[%k, %j] : memref<4x4xf32>

      // Scale Output Row (The inverse)
      %out_val = memref.load %output[%k, %j] : memref<4x4xf32>
      %scaled_out = arith.divf %out_val, %pivot_val : f32
      memref.store %scaled_out, %output[%k, %j] : memref<4x4xf32>
    }

    // --- Step B: Eliminate other rows ---
    scf.for %i = %c0 to %c4 step %c1 {
      %is_not_pivot = arith.cmpi ne, %i, %k : index
      scf.if %is_not_pivot {
        %factor = memref.load %working_input[%i, %k] : memref<4x4xf32>

        scf.for %j = %c0 to %c4 step %c1 {
          // Update Input matrix
          %in_k = memref.load %working_input[%k, %j] : memref<4x4xf32>
          %in_i = memref.load %working_input[%i, %j] : memref<4x4xf32>
          %sub_in = arith.mulf %factor, %in_k : f32
          %new_in = arith.subf %in_i, %sub_in : f32
          memref.store %new_in, %working_input[%i, %j] : memref<4x4xf32>

          // Update Output matrix
          %out_k = memref.load %output[%k, %j] : memref<4x4xf32>
          %out_i = memref.load %output[%i, %j] : memref<4x4xf32>
          %sub_out = arith.mulf %factor, %out_k : f32
          %new_out = arith.subf %out_i, %sub_out : f32
          memref.store %new_out, %output[%i, %j] : memref<4x4xf32>
        }
      }
    }
  }

  // Deallocate the temporary working copy before returning
  memref.dealloc %working_input : memref<4x4xf32>

  return %output : memref<4x4xf32>
}

func.func private @dot(%A: memref<4x4xf32>, %B: memref<4x4xf32>) -> memref<4x4xf32> {
    return %A : memref<4x4xf32>
}

// Main now treats memory entirely as SSA values being passed around
func.func @main(%A: memref<4x4xf32>, %B: memref<4x4xf32>) -> memref<4x4xf32> {
  // 1. Get the inverse (allocates its own memory and returns it)
  %inv = func.call @get_inv(%A) : (memref<4x4xf32>) -> memref<4x4xf32>

  // 2. Compute dot product (allocates its own Result memory and returns it)
  %result = func.call @dot(%inv, %B) : (memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>

  // 3. Since main now "owns" %inv, we must deallocate it to prevent memory leaks
  memref.dealloc %inv : memref<4x4xf32>

  return %inv : memref<4x4xf32>
}

  pdl_interp.func @matcher(%arg0 : !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb0, ^bb40
  ^bb1:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb2, ^bb1
  ^bb2:
    %0 = pdl_interp.get_operand 0 of %arg0
    pdl_interp.is_not_null %0 : !pdl.value -> ^bb3, ^bb1
  ^bb3:
    %1 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %1 : !pdl.value -> ^bb4, ^bb1
  ^bb4:
    %2 = ematch.get_class_result %1
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5, ^bb1
  ^bb5:
    %3 = ematch.get_class_vals %0
    pdl_interp.foreach %4 : !pdl.value in %3 {
      %5 = pdl_interp.get_defining_op of %4 : !pdl.value
      pdl_interp.is_not_null %5 : !pdl.operation -> ^bb6, ^bb7
    ^bb7:
      pdl_interp.continue
    ^bb6:
      pdl_interp.check_operation_name of %5 is "arith.constant" -> ^bb8, ^bb7
    ^bb8:
      pdl_interp.check_operand_count of %5 is 0 -> ^bb9, ^bb7
    ^bb9:
      pdl_interp.check_result_count of %5 is 1 -> ^bb10, ^bb7
    ^bb10:
      %6 = pdl_interp.get_attribute "value" of %5
      pdl_interp.is_not_null %6 : !pdl.attribute -> ^bb11, ^bb7
    ^bb11:
      %7 = pdl_interp.create_attribute 1 : i1
      pdl_interp.are_equal %6, %7 : !pdl.attribute -> ^bb12, ^bb16
    ^bb12:
      %8 = pdl_interp.get_result 0 of %5
      pdl_interp.is_not_null %8 : !pdl.value -> ^bb13, ^bb7
    ^bb13:
      %9 = ematch.get_class_result %8
      pdl_interp.is_not_null %9 : !pdl.value -> ^bb14, ^bb7
    ^bb14:
      pdl_interp.are_equal %9, %0 : !pdl.value -> ^bb15, ^bb7
    ^bb15:
      %10 = pdl_interp.get_value_type of %2 : !pdl.type
      pdl_interp.record_match @rewriters::@if_true_rewriter(%arg0, %10 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("scf.if") -> ^bb7
    ^bb16:
      %11 = pdl_interp.create_attribute 0 : i1
      pdl_interp.are_equal %6, %11 : !pdl.attribute -> ^bb17, ^bb7
    ^bb17:
      %12 = pdl_interp.get_result 0 of %5
      pdl_interp.is_not_null %12 : !pdl.value -> ^bb18, ^bb7
    ^bb18:
      %13 = ematch.get_class_result %12
      pdl_interp.is_not_null %13 : !pdl.value -> ^bb19, ^bb7
    ^bb19:
      pdl_interp.are_equal %13, %0 : !pdl.value -> ^bb20, ^bb7
    ^bb20:
      %14 = pdl_interp.get_value_type of %2 : !pdl.type
      pdl_interp.record_match @rewriters::@if_false_rewriter(%arg0, %14 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("scf.if") -> ^bb7
    } -> ^bb1
    ^bb40:
      pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb41, ^bb50
    ^bb41:
      pdl_interp.check_result_count of %arg0 is 1 -> ^bb100, ^bb1
    ^bb100:
      %100 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb42, ^bb1
    ^bb42:
      %110 = pdl_interp.get_attribute "sym_visibility" of %100
      %112 = pdl_interp.create_attribute "private"
      pdl_interp.are_equal %110, %112 : !pdl.attribute -> ^bb1, ^bb48
    ^bb48:
      %101 = pdl_interp_region.get_region 0 of %100 : !pdl_region.region
      %102 = pdl_interp.apply_constraint "replace_return_with_yield"(%101 : !pdl_region.region) : !pdl_region.region -> ^bb43, ^bb1
    ^bb43:
      %103 = pdl_interp.apply_constraint "get_arguments_of_function"(%100 : !pdl.operation) : !pdl.range<value> -> ^bb46, ^bb1
    ^bb46:
      %104 = pdl_interp.get_result 0 of %arg0
      %105 = pdl_interp.get_value_type of %104 : !pdl.type
      %106 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%102 : !pdl_region.region) -> (%105 : !pdl.type)
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%106, %100, %arg0 : !pdl.operation, !pdl.operation, !pdl.operation) -> ^bb45, ^bb1
    ^bb45:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %106 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
    ^bb50:
      pdl_interp.check_operation_name of %arg0 is "math.exp" -> ^bb51, ^bb1
    ^bb51:
      pdl_interp.check_operand_count of %arg0 is 1 -> ^bb52, ^bb1
    ^bb52:
      %200 = pdl_interp.get_operand 0 of %arg0
      pdl_interp.is_not_null %200 : !pdl.value -> ^bb53, ^bb1
    ^bb53:
      %201 = pdl_interp.get_defining_op of %200 : !pdl.value
      pdl_interp.is_not_null %201 : !pdl.operation -> ^bb54, ^bb1
    ^bb54:
      pdl_interp.check_operation_name of %201 is "arith.addf" -> ^bb55, ^bb1
    ^bb55:
      %202 = pdl_interp.get_operand 0 of %201
      %204 = ematch.get_class_vals %202
      pdl_interp.foreach %205 : !pdl.value in %204 {
        %206 = pdl_interp.get_defining_op of %205 : !pdl.value
        pdl_interp.is_not_null %206 : !pdl.operation -> ^bb57, ^bb56
        ^bb56:
          pdl_interp.continue
        ^bb57:
          pdl_interp.check_operation_name of %206 is "math.log" -> ^bb58, ^bb56
        ^bb58:
          %207 = pdl_interp.get_operand 1 of %201
          %208 = ematch.get_class_vals %207
          pdl_interp.foreach %209 : !pdl.value in %208 {
            %210 = pdl_interp.get_defining_op of %209 : !pdl.value
            pdl_interp.is_not_null %210 : !pdl.operation -> ^bb60, ^bb59
            ^bb59:
              pdl_interp.continue
            ^bb60:
              pdl_interp.check_operation_name of %210 is "math.log" -> ^bb61, ^bb59
            ^bb61:
              %211 = pdl_interp.get_result 0 of %arg0
              %212 = ematch.get_class_result %211
              %213 = pdl_interp.get_value_type of %212 : !pdl.type
              pdl_interp.finalize
          } -> ^bb56
      } -> ^bb1
   }

  builtin.module @rewriters {
    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

     pdl_interp.func @if_false_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
      %0 = pdl_interp.get_result 0 of %arg1
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    }
  }
