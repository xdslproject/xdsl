 func.func @ex4(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = arith.constant 0.000000e+00 : f64
    %c0_0 = arith.constant 0 : index
    %1 = arith.constant 0.000000e+00 : f64
    %c0_1 = arith.constant 0 : index
    %2 = arith.constant 0.000000e+00 : f64
    %c0_2 = arith.constant 0 : index
    %3 = arith.constant 0.000000e+00 : f64
    %c0_3 = arith.constant 0 : index
    %4 = arith.constant 0.000000e+00 : f64
    %c0_4 = arith.constant 0 : index
    %5 = arith.constant 0.000000e+00 : f64
    %c0_5 = arith.constant 0 : index
    %6 = arith.constant 0.000000e+00 : f64
    %c0_6 = arith.constant 0 : index
    %7 = arith.constant 0.000000e+00 : f64
    %c0_7 = arith.constant 0 : index
    %8 = arith.constant 0.000000e+00 : f64
    %c0_8 = arith.constant 0 : index
    %9 = arith.constant 0.000000e+00 : f64
    %c0_9 = arith.constant 0 : index
    %10 = arith.constant 0.000000e+00 : f64
    %c0_10 = arith.constant 0 : index
    %11 = arith.constant 0.000000e+00 : f64
    %c0_11 = arith.constant 0 : index
    %12 = arith.constant 0.000000e+00 : f64
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c0_14 = arith.constant 0 : index
    %c0_15 = arith.constant 0 : index
    %c0_16 = arith.constant 0 : index
    %true = arith.constant true
    %13 = arith.constant 0.000000e+00 : f64
    %14 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %cst = arith.constant 3.1415926535900001 : f64
            %c0_17 = arith.constant 0 : index
            scf.yield %cst : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %7 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %7 : f64
    }
    %15 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %c0_18 = arith.constant 0 : index
            scf.yield %arg1 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %6 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %6 : f64
    }
    %16 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %cst = arith.constant 2.000000e+00 : f64
            %c0_17 = arith.constant 0 : index
            %26 = arith.mulf %cst, %14 : f64
            %c0_18 = arith.constant 0 : index
            %27 = arith.mulf %26, %arg2 : f64
            %c0_19 = arith.constant 0 : index
            %28 = arith.mulf %27, %arg3 : f64
            %c0_20 = arith.constant 0 : index
            scf.yield %28 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %5 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %5 : f64
    }
    %17 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %c0_18 = arith.constant 0 : index
            %26 = arith.mulf %15, %15 : f64
            %c0_19 = arith.constant 0 : index
            %c0_20 = arith.constant 0 : index
            %27 = arith.mulf %16, %16 : f64
            %28 = arith.addf %26, %27 : f64
            %c0_21 = arith.constant 0 : index
            scf.yield %28 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %4 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %4 : f64
    }
    %18 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %c0_18 = arith.constant 0 : index
            %26 = arith.mulf %arg4, %15 : f64
            %c0_19 = arith.constant 0 : index
            %27 = arith.divf %26, %17 : f64
            %c0_20 = arith.constant 0 : index
            scf.yield %27 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %3 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %3 : f64
    }
    %19 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %c0_18 = arith.constant 0 : index
            %26 = arith.mulf %arg4, %16 : f64
            %27 = arith.negf %26 : f64
            %c0_19 = arith.constant 0 : index
            %28 = arith.divf %27, %17 : f64
            %c0_20 = arith.constant 0 : index
            scf.yield %28 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %2 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %2 : f64
    }
    %20 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %c0_18 = arith.constant 0 : index
            %26 = arith.mulf %18, %18 : f64
            %c0_19 = arith.constant 0 : index
            %c0_20 = arith.constant 0 : index
            %27 = arith.mulf %19, %19 : f64
            %28 = arith.addf %26, %27 : f64
            %29 = math.sqrt %28 : f64
            %c0_21 = arith.constant 0 : index
            scf.yield %29 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %1 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %1 : f64
    }
    %21 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %c0_18 = arith.constant 0 : index
            %26 = arith.divf %19, %18 : f64
            %27 = math.atan %26 : f64
            %c0_19 = arith.constant 0 : index
            scf.yield %27 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %0 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %0 : f64
    }
    %22 = scf.if %true -> (f64) {
      %23 = scf.execute_region -> f64 {
        %24 = scf.if %true -> (f64) {
          %25 = scf.execute_region -> f64 {
            %c0_17 = arith.constant 0 : index
            %cst = arith.constant 2.000000e+00 : f64
            %c0_18 = arith.constant 0 : index
            %26 = arith.mulf %cst, %14 : f64
            %c0_19 = arith.constant 0 : index
            %27 = arith.mulf %26, %arg2 : f64
            %c0_20 = arith.constant 0 : index
            %28 = arith.mulf %27, %arg0 : f64
            %c0_21 = arith.constant 0 : index
            %29 = arith.addf %28, %21 : f64
            %30 = math.cos %29 : f64
            %31 = arith.mulf %20, %30 : f64
            %false = arith.constant false
            scf.yield %31 : f64
          }
          scf.yield %25 : f64
        } else {
          scf.yield %13 : f64
        }
        scf.yield %24 : f64
      }
      scf.yield %23 : f64
    } else {
      scf.yield %13 : f64
    }
    return %22 : f64
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