  func.func @is_prime(%arg0: i64) -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %c1_i8 = arith.constant 1 : i8
    %c1_i64 = arith.constant 1 : i64
    %c0_i8 = arith.constant 0 : i8
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %0 = arith.constant 0 : i8
    cf.br ^bb1(%c2_i64, %true, %0, %true : i64, i1, i8, i1)
  ^bb1(%1: i64, %2: i1, %3: i8, %4: i1):  // 2 preds: ^bb0, ^bb2
    %5 = arith.uitofp %1 : i64 to f64
    %6 = arith.uitofp %arg0 : i64 to f64
    %7 = math.sqrt %6 : f64
    %8 = arith.addf %7, %cst : f64
    %9 = arith.cmpf olt, %5, %8 : f64
    %10 = arith.andi %9, %4 : i1
    cf.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = arith.remui %arg0, %1 : i64
    %12 = arith.cmpi eq, %11, %c0_i64 : i64
    %13 = arith.cmpi ne, %11, %c0_i64 : i64
    %14 = arith.andi %13, %2 : i1
    %15 = arith.select %12, %c0_i8, %3 : i8
    %16 = arith.cmpi ne, %11, %c0_i64 : i64
    %17 = arith.xori %4, %true : i1
    %18 = arith.andi %4, %14 : i1
    %19 = arith.andi %17, %2 : i1
    %20 = arith.ori %18, %19 : i1
    %21 = arith.select %4, %15, %3 : i8
    %22 = arith.andi %4, %16 : i1
    %23 = scf.if %22 -> (i64) {
      %25 = arith.addi %1, %c1_i64 : i64
      scf.yield %25 : i64
    } else {
      scf.yield %1 : i64
    }
    cf.br ^bb1(%23, %20, %21, %22 : i64, i1, i8, i1)
  ^bb3:  // pred: ^bb1
    %24 = arith.select %2, %c1_i8, %3 : i8
    return %24 : i8
  }
  func.func @sum_of_primes(%arg0: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i8 = arith.constant 0 : i8
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    cf.br ^bb1(%c3_i64, %c2_i64 : i64, i64)
  ^bb1(%0: i64, %1: i64):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi ult, %0, %arg0 : i64
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %3 = call @is_prime(%0) : (i64) -> i8
    %4 = arith.cmpi ne, %3, %c0_i8 : i8
    %5 = scf.if %4 -> (i64) {
      %7 = arith.addi %1, %0 : i64
      scf.yield %7 : i64
    } else {
      scf.yield %1 : i64
    }
    %6 = arith.addi %0, %c2_i64 : i64
    cf.br ^bb1(%6, %5 : i64, i64)
  ^bb3:  // pred: ^bb1
    return %1 : i64
  }



pdl_interp.func @matcher(%arg0: !pdl.operation) {
  // check if operation is an if statement
  pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb1, ^bb6
^bb1:
  %m1 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb2, ^bb30
^bb2:
  %m2 = pdl_interp_region.get_region 0 of %m1 : !pdl_region.region
  %m3 = pdl_interp.apply_constraint "replace_return_with_yield"(%m2 : !pdl_region.region) : !pdl_region.region -> ^bb3, ^bb30
^bb3:
  %m4 = pdl_interp.apply_constraint "get_arguments_of_function"(%m1 : !pdl.operation) : !pdl.range<value> -> ^bb4, ^bb30
^bb4:
  %m10 = pdl_interp.get_result 0 of %arg0
  %m11 = pdl_interp.get_value_type of %m10 : !pdl.type
  %m12 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%m3 : !pdl_region.region) -> (%m11 : !pdl.type)
  pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%m12, %m4 : !pdl.operation, !pdl.range<value>) : ^bb5, ^bb30
^bb5:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0, %m12 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb30
^bb6:
  pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb7, ^bb10
^bb7:
  %m20 = pdl_interp.get_operand 0 of %arg0
  %m21 = pdl_interp.get_defining_op of %m20 : !pdl.value
  pdl_interp.check_operation_name of %m21 is "arith.constant" -> ^bb8, ^bb10
^bb8:
  %m22 = pdl_interp.get_attribute "value" of %m21
  %m23 = pdl_interp.create_attribute 1 : i1
  pdl_interp.are_equal %m22, %m23 : !pdl.attribute -> ^bb9, ^bb10
^bb9:
  %m24 = pdl_interp.get_result 0 of %arg0
  %m25 = pdl_interp.get_value_type of %m24 : !pdl.type
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_2(%arg0 , %m25 : !pdl.operation, !pdl.type) : benefit(1) -> ^bb10
^bb10:
  pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb11, ^bb30
^bb11:
  %r000 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
  pdl_interp.apply_constraint "at_most_1_block"(%r000: !pdl_region.region) : ^bb12, ^bb30
^bb12:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_3(%arg0 : !pdl.operation) : benefit(1) -> ^bb30
^bb30:
  pdl_interp.finalize
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0 : !pdl.operation, %r1 : !pdl.operation) {
        %r3 = pdl_interp.get_result 0 of %r1
        pdl_interp.replace %arg0 with (%r3 : !pdl.value)
        pdl_interp.finalize
    }

    pdl_interp.func @pdl_generated_rewriter_2(%arg0: !pdl.operation, %arg1 : !pdl.type) {
      %r0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %r1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%r0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %r2 = pdl_interp.get_result 0 of %r1
      pdl_interp.replace %arg0 with (%r2 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @pdl_generated_rewriter_3(%arg0: !pdl.operation) {
      %r000 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %r1 = pdl_interp_region.inline_region %arg0 with (%r000 : !pdl_region.region)
      pdl_interp.replace %arg0 with (%r1 : !pdl.value)
      pdl_interp.finalize
    }
}