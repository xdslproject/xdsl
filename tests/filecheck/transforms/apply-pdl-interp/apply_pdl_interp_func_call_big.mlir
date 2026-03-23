  func.func @ex0(%arg0: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = arith.constant 0.000000e+00 : f64
    %c0_0 = arith.constant 0 : index
    %1 = arith.constant 0.000000e+00 : f64
    %c0_1 = arith.constant 0 : index
    %2 = arith.constant 0 : i32
    %c0_2 = arith.constant 0 : index
    %3 = arith.constant 0.000000e+00 : f64
    %c0_3 = arith.constant 0 : index
    %4 = arith.constant 0.000000e+00 : f64
    %c0_4 = arith.constant 0 : index
    %5 = arith.constant 0.000000e+00 : f64
    %c0_5 = arith.constant 0 : index
    %true = arith.constant true
    %6 = arith.constant 0.000000e+00 : f64
    %7 = scf.if %true -> (f64) {
      %12 = scf.execute_region -> f64 {
        %13 = scf.if %true -> (f64) {
          %14 = scf.execute_region -> f64 {
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            scf.yield %arg0 : f64
          }
          scf.yield %14 : f64
        } else {
          scf.yield %4 : f64
        }
        scf.yield %13 : f64
      }
      scf.yield %12 : f64
    } else {
      scf.yield %4 : f64
    }
    %8 = scf.if %true -> (f64) {
      %12 = scf.execute_region -> f64 {
        %13 = scf.if %true -> (f64) {
          %14 = scf.execute_region -> f64 {
            %cst = arith.constant 0.000000e+00 : f64
            %c0_6 = arith.constant 0 : index
            scf.yield %cst : f64
          }
          scf.yield %14 : f64
        } else {
          scf.yield %3 : f64
        }
        scf.yield %13 : f64
      }
      scf.yield %12 : f64
    } else {
      scf.yield %3 : f64
    }
    %9 = scf.if %true -> (i32) {
      %12 = scf.execute_region -> i32 {
        %13 = scf.if %true -> (i32) {
          %14 = scf.execute_region -> i32 {
            %c0_6 = arith.constant 0 : index
            %cst = arith.constant 1.000000e+01 : f64
            %15 = arith.cmpf olt, %8, %cst : f64
            %16 = arith.extui %15 : i1 to i32
            %c0_7 = arith.constant 0 : index
            scf.yield %16 : i32
          }
          scf.yield %14 : i32
        } else {
          scf.yield %2 : i32
        }
        scf.yield %13 : i32
      }
      scf.yield %12 : i32
    } else {
      scf.yield %2 : i32
    }
    %10 = scf.if %true -> (f64) {
      %12 = scf.execute_region -> f64 {
        %13 = scf.if %true -> (f64) {
          %14 = scf.execute_region -> f64 {
            %true_6 = arith.constant true
            cf.br ^bb1(%0, %1, %9, %8, %7 : f64, f64, i32, f64, f64)
          ^bb1(%15: f64, %16: f64, %17: i32, %18: f64, %19: f64):  // 2 preds: ^bb0, ^bb2
            %c0_7 = arith.constant 0 : index
            %c0_i32 = arith.constant 0 : i32
            %20 = arith.cmpi ne, %17, %c0_i32 : i32
            %21 = arith.andi %20, %true_6 : i1
            cf.cond_br %21, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %22 = scf.if %true_6 -> (f64) {
              %27 = scf.execute_region -> f64 {
                %28 = scf.if %true_6 -> (f64) {
                  %29 = scf.execute_region -> f64 {
                    %c0_8 = arith.constant 0 : index
                    %c0_9 = arith.constant 0 : index
                    %c0_10 = arith.constant 0 : index
                    %cst = arith.constant 3.000000e+00 : f64
                    %30 = math.powf %19, %cst : f64
                    %cst_11 = arith.constant 6.000000e+00 : f64
                    %31 = arith.divf %30, %cst_11 : f64
                    %32 = arith.subf %19, %31 : f64
                    %c0_12 = arith.constant 0 : index
                    %cst_13 = arith.constant 5.000000e+00 : f64
                    %33 = math.powf %19, %cst_13 : f64
                    %cst_14 = arith.constant 1.200000e+02 : f64
                    %34 = arith.divf %33, %cst_14 : f64
                    %35 = arith.addf %32, %34 : f64
                    %c0_15 = arith.constant 0 : index
                    %cst_16 = arith.constant 7.000000e+00 : f64
                    %36 = math.powf %19, %cst_16 : f64
                    %cst_17 = arith.constant 5.040000e+03 : f64
                    %37 = arith.divf %36, %cst_17 : f64
                    %38 = arith.addf %35, %37 : f64
                    %cst_18 = arith.constant 1.000000e+00 : f64
                    %c0_19 = arith.constant 0 : index
                    %c0_20 = arith.constant 0 : index
                    %39 = arith.mulf %19, %19 : f64
                    %cst_21 = arith.constant 2.000000e+00 : f64
                    %40 = arith.divf %39, %cst_21 : f64
                    %41 = arith.subf %cst_18, %40 : f64
                    %c0_22 = arith.constant 0 : index
                    %cst_23 = arith.constant 4.000000e+00 : f64
                    %42 = math.powf %19, %cst_23 : f64
                    %cst_24 = arith.constant 2.400000e+01 : f64
                    %43 = arith.divf %42, %cst_24 : f64
                    %44 = arith.addf %41, %43 : f64
                    %c0_25 = arith.constant 0 : index
                    %cst_26 = arith.constant 6.000000e+00 : f64
                    %45 = math.powf %19, %cst_26 : f64
                    %cst_27 = arith.constant 7.200000e+02 : f64
                    %46 = arith.divf %45, %cst_27 : f64
                    %47 = arith.addf %44, %46 : f64
                    %48 = arith.divf %38, %47 : f64
                    %49 = arith.subf %19, %48 : f64
                    %c0_28 = arith.constant 0 : index
                    scf.yield %49 : f64
                  }
                  scf.yield %29 : f64
                } else {
                  scf.yield %16 : f64
                }
                scf.yield %28 : f64
              }
              scf.yield %27 : f64
            } else {
              scf.yield %16 : f64
            }
            %23 = scf.if %true_6 -> (f64) {
              %27 = scf.execute_region -> f64 {
                %28 = scf.if %true_6 -> (f64) {
                  %29 = scf.execute_region -> f64 {
                    %c0_8 = arith.constant 0 : index
                    %cst = arith.constant 1.000000e+00 : f64
                    %30 = arith.addf %18, %cst : f64
                    %c0_9 = arith.constant 0 : index
                    scf.yield %30 : f64
                  }
                  scf.yield %29 : f64
                } else {
                  scf.yield %15 : f64
                }
                scf.yield %28 : f64
              }
              scf.yield %27 : f64
            } else {
              scf.yield %15 : f64
            }
            %24 = scf.if %true_6 -> (f64) {
              scf.execute_region {
                %c0_8 = arith.constant 0 : index
                %c0_9 = arith.constant 0 : index
                scf.yield
              }
              scf.yield %22 : f64
            } else {
              scf.yield %19 : f64
            }
            %25 = scf.if %true_6 -> (f64) {
              scf.execute_region {
                %c0_8 = arith.constant 0 : index
                %c0_9 = arith.constant 0 : index
                scf.yield
              }
              scf.yield %23 : f64
            } else {
              scf.yield %18 : f64
            }
            %26 = scf.if %true_6 -> (i32) {
              %27 = scf.execute_region -> i32 {
                %c0_8 = arith.constant 0 : index
                %cst = arith.constant 1.000000e+01 : f64
                %28 = arith.cmpf olt, %25, %cst : f64
                %29 = arith.extui %28 : i1 to i32
                %c0_9 = arith.constant 0 : index
                scf.yield %29 : i32
              }
              scf.yield %27 : i32
            } else {
              scf.yield %17 : i32
            }
            cf.br ^bb1(%23, %22, %26, %25, %24 : f64, f64, i32, f64, f64)
          ^bb3:  // pred: ^bb1
            scf.yield %19 : f64
          }
          scf.yield %14 : f64
        } else {
          scf.yield %7 : f64
        }
        scf.yield %13 : f64
      }
      scf.yield %12 : f64
    } else {
      scf.yield %7 : f64
    }
    %11 = scf.if %true -> (f64) {
      %12 = scf.execute_region -> f64 {
        %13 = scf.if %true -> (f64) {
          scf.execute_region {
            %c0_6 = arith.constant 0 : index
            %false = arith.constant false
            scf.yield
          }
          scf.yield %10 : f64
        } else {
          scf.yield %6 : f64
        }
        scf.yield %13 : f64
      }
      scf.yield %12 : f64
    } else {
      scf.yield %6 : f64
    }
    return %11 : f64
  }
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = arith.constant 0 : i32
    %c0_0 = arith.constant 0 : index
    %1 = arith.constant 0.000000e+00 : f32
    %c0_1 = arith.constant 0 : index
    %2 = arith.constant 0 : i32
    %c0_2 = arith.constant 0 : index
    %3 = arith.constant 0 : i32
    %true = arith.constant true
    %4 = arith.constant 0 : i32
    %5 = scf.if %true -> (i32) {
      %8 = scf.execute_region -> i32 {
        %9 = scf.if %true -> (i32) {
          %10 = scf.execute_region -> i32 {
            %c0_i32 = arith.constant 0 : i32
            %c0_3 = arith.constant 0 : index
            scf.yield %c0_i32 : i32
          }
          scf.yield %10 : i32
        } else {
          scf.yield %3 : i32
        }
        scf.yield %9 : i32
      }
      scf.yield %8 : i32
    } else {
      scf.yield %3 : i32
    }
    %6 = scf.if %true -> (i32) {
      %8 = scf.execute_region -> i32 {
        %9 = scf.if %true -> (i32) {
          %10 = scf.execute_region -> i32 {
            %11 = scf.if %true -> (i32) {
              %23 = scf.execute_region -> i32 {
                %c0_i32_5 = arith.constant 0 : i32
                %c0_6 = arith.constant 0 : index
                scf.yield %c0_i32_5 : i32
              }
              scf.yield %23 : i32
            } else {
              scf.yield %2 : i32
            }
            %true_3 = arith.constant true
            cf.br ^bb1(%0, %1, %11, %5 : i32, f32, i32, i32)
          ^bb1(%12: i32, %13: f32, %14: i32, %15: i32):  // 2 preds: ^bb0, ^bb2
            %c0_4 = arith.constant 0 : index
            %c1999_i32 = arith.constant 1999 : i32
            %16 = arith.cmpi sle, %14, %c1999_i32 : i32
            %17 = arith.extui %16 : i1 to i32
            %c0_i32 = arith.constant 0 : i32
            %18 = arith.cmpi ne, %17, %c0_i32 : i32
            %19 = arith.andi %18, %true_3 : i1
            cf.cond_br %19, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %20 = scf.if %true_3 -> (f32) {
              %23 = scf.execute_region -> f32 {
                %24 = scf.if %true_3 -> (f32) {
                  %25 = scf.execute_region -> f32 {
                    %cst = arith.constant 9.990000e-01 : f32
                    %cst_5 = arith.constant -9.990000e-01 : f32
                    %c0_6 = arith.constant 0 : index
                    %26 = arith.sitofp %14 : i32 to f32
                    %cst_7 = arith.constant 1.000000e-03 : f32
                    %27 = arith.mulf %26, %cst_7 : f32
                    %28 = arith.addf %cst_5, %27 : f32
                    %c0_8 = arith.constant 0 : index
                    scf.yield %28 : f32
                  }
                  scf.yield %25 : f32
                } else {
                  scf.yield %13 : f32
                }
                scf.yield %24 : f32
              }
              scf.yield %23 : f32
            } else {
              scf.yield %13 : f32
            }
            %21:2 = scf.if %true_3 -> (i32, i32) {
              %23:2 = scf.execute_region -> (i32, i32) {
                %24:2 = scf.if %true_3 -> (i32, i32) {
                  %25:2 = scf.execute_region -> (i32, i32) {
                    %26 = scf.if %true_3 -> (i32) {
                      %35 = scf.execute_region -> i32 {
                        %c0_i32_8 = arith.constant 0 : i32
                        %c0_9 = arith.constant 0 : index
                        scf.yield %c0_i32_8 : i32
                      }
                      scf.yield %35 : i32
                    } else {
                      scf.yield %12 : i32
                    }
                    %true_5 = arith.constant true
                    cf.br ^bb1(%26, %15 : i32, i32)
                  ^bb1(%27: i32, %28: i32):  // 2 preds: ^bb0, ^bb2
                    %c0_6 = arith.constant 0 : index
                    %c10000_i32 = arith.constant 10000 : i32
                    %29 = arith.cmpi slt, %27, %c10000_i32 : i32
                    %30 = arith.extui %29 : i1 to i32
                    %c0_i32_7 = arith.constant 0 : i32
                    %31 = arith.cmpi ne, %30, %c0_i32_7 : i32
                    %32 = arith.andi %31, %true_5 : i1
                    cf.cond_br %32, ^bb2, ^bb3
                  ^bb2:  // pred: ^bb1
                    %33 = scf.if %true_5 -> (i32) {
                      %35 = scf.execute_region -> i32 {
                        %c0_8 = arith.constant 0 : index
                        %36 = arith.extf %20 : f32 to f64
                        %37 = func.call @ex0(%36) : (f64) -> f64
                        %38 = arith.fptosi %37 : f64 to i32
                        %c0_9 = arith.constant 0 : index
                        %39 = arith.addi %28, %38 : i32
                        %c0_10 = arith.constant 0 : index
                        scf.yield %39 : i32
                      }
                      scf.yield %35 : i32
                    } else {
                      scf.yield %28 : i32
                    }
                    %34 = scf.if %true_5 -> (i32) {
                      %35 = scf.execute_region -> i32 {
                        %c0_8 = arith.constant 0 : index
                        %c1_i32 = arith.constant 1 : i32
                        %36 = arith.addi %27, %c1_i32 : i32
                        %c0_9 = arith.constant 0 : index
                        scf.yield %36 : i32
                      }
                      scf.yield %35 : i32
                    } else {
                      scf.yield %27 : i32
                    }
                    cf.br ^bb1(%34, %33 : i32, i32)
                  ^bb3:  // pred: ^bb1
                    scf.yield %27, %28 : i32, i32
                  }
                  scf.yield %25#0, %25#1 : i32, i32
                } else {
                  scf.yield %12, %15 : i32, i32
                }
                scf.yield %24#0, %24#1 : i32, i32
              }
              scf.yield %23#0, %23#1 : i32, i32
            } else {
              scf.yield %12, %15 : i32, i32
            }
            %22 = scf.if %true_3 -> (i32) {
              %23 = scf.execute_region -> i32 {
                %c0_5 = arith.constant 0 : index
                %c1_i32 = arith.constant 1 : i32
                %24 = arith.addi %14, %c1_i32 : i32
                %c0_6 = arith.constant 0 : index
                scf.yield %24 : i32
              }
              scf.yield %23 : i32
            } else {
              scf.yield %14 : i32
            }
            cf.br ^bb1(%21#0, %20, %22, %21#1 : i32, f32, i32, i32)
          ^bb3:  // pred: ^bb1
            scf.yield %15 : i32
          }
          scf.yield %10 : i32
        } else {
          scf.yield %5 : i32
        }
        scf.yield %9 : i32
      }
      scf.yield %8 : i32
    } else {
      scf.yield %5 : i32
    }
    %7 = scf.if %true -> (i32) {
      %8 = scf.execute_region -> i32 {
        %9 = scf.if %true -> (i32) {
          scf.execute_region {
            %c0_3 = arith.constant 0 : index
            %false = arith.constant false
            scf.yield
          }
          scf.yield %6 : i32
        } else {
          scf.yield %4 : i32
        }
        scf.yield %9 : i32
      }
      scf.yield %8 : i32
    } else {
      scf.yield %4 : i32
    }
    return %7 : i32
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