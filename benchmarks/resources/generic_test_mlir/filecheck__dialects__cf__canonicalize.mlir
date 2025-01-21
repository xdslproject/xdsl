"builtin.module"() ({
  "func.func"() <{function_type = () -> i1, sym_name = "assert_true"}> ({
    %41 = "arith.constant"() <{value = true}> : () -> i1
    "cf.assert"(%41) <{msg = "assert true"}> : (i1) -> ()
    "func.return"(%41) : (i1) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "br_folding"}> ({
    %39 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "cf.br"(%39)[^bb1] : (i32) -> ()
  ^bb1(%40: i32):  // pred: ^bb0
    "func.return"(%40) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, i32) -> (i32, i32), sym_name = "br_passthrough"}> ({
  ^bb0(%arg44: i32, %arg45: i32):
    "test.termop"()[^bb1, ^bb2, ^bb3] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg44)[^bb2] : (i32) -> ()
  ^bb2(%36: i32):  // 2 preds: ^bb0, ^bb1
    "cf.br"(%36, %arg45)[^bb3] : (i32, i32) -> ()
  ^bb3(%37: i32, %38: i32):  // 2 preds: ^bb0, ^bb2
    "func.return"(%37, %38) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "br_dead_passthrough"}> ({
    "cf.br"()[^bb2] : () -> ()
  ^bb1:  // no predecessors
    "cf.br"()[^bb2] : () -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1, i32) -> (), sym_name = "cond_br_folding"}> ({
  ^bb0(%arg42: i1, %arg43: i32):
    %33 = "arith.constant"() <{value = false}> : () -> i1
    %34 = "arith.constant"() <{value = true}> : () -> i1
    "cf.cond_br"(%arg42, %arg43)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
  ^bb1:  // pred: ^bb0
    "cf.cond_br"(%34, %arg43)[^bb3, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
  ^bb2(%35: i32):  // 3 preds: ^bb0, ^bb1, ^bb2
    "cf.cond_br"(%33, %35)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (i1, i32) -> ()
  ^bb3:  // 2 preds: ^bb1, ^bb2
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> (), sym_name = "cond_br_and_br_folding"}> ({
  ^bb0(%arg41: i32):
    %30 = "arith.constant"() <{value = false}> : () -> i1
    %31 = "arith.constant"() <{value = true}> : () -> i1
    "cf.cond_br"(%31, %arg41)[^bb2, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
  ^bb1(%32: i32):  // 2 preds: ^bb0, ^bb1
    "cf.cond_br"(%30, %32)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (i1, i32) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, i32, i32, i1) -> (i32, i32), sym_name = "cond_br_passthrough"}> ({
  ^bb0(%arg37: i32, %arg38: i32, %arg39: i32, %arg40: i1):
    "cf.cond_br"(%arg40, %arg37, %arg39, %arg39)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 1, 2>}> : (i1, i32, i32, i32) -> ()
  ^bb1(%27: i32):  // pred: ^bb0
    "cf.br"(%27, %arg38)[^bb2] : (i32, i32) -> ()
  ^bb2(%28: i32, %29: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%28, %29) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1) -> (), sym_name = "cond_br_pass_through_fail"}> ({
  ^bb0(%arg36: i1):
    "cf.cond_br"(%arg36)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "test.op"() : () -> ()
    "cf.br"()[^bb2] : () -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1, i32) -> (), sym_name = "cond_br_same_successor"}> ({
  ^bb0(%arg34: i1, %arg35: i32):
    "cf.cond_br"(%arg34, %arg35, %arg35)[^bb1, ^bb1] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, i32) -> ()
  ^bb1(%26: i32):  // 2 preds: ^bb0, ^bb0
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1, i32, i32, tensor<2xi32>, tensor<2xi32>) -> (i32, tensor<2xi32>), sym_name = "cond_br_same_successor_insert_select"}> ({
  ^bb0(%arg29: i1, %arg30: i32, %arg31: i32, %arg32: tensor<2xi32>, %arg33: tensor<2xi32>):
    "cf.cond_br"(%arg29, %arg30, %arg32, %arg31, %arg33)[^bb1, ^bb1] <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (i1, i32, tensor<2xi32>, i32, tensor<2xi32>) -> ()
  ^bb1(%24: i32, %25: tensor<2xi32>):  // 2 preds: ^bb0, ^bb0
    "func.return"(%24, %25) : (i32, tensor<2xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1) -> (), sym_name = "cond_br_from_cond_br_with_same_condition"}> ({
  ^bb0(%arg28: i1):
    "cf.cond_br"(%arg28)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.cond_br"(%arg28)[^bb3, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "test.termop"() : () -> ()
  ^bb3:  // pred: ^bb1
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1) -> (), sym_name = "branchCondProp"}> ({
  ^bb0(%arg27: i1):
    "cf.cond_br"(%arg27)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "test.op"(%arg27) : (i1) -> ()
    "cf.br"()[^bb3] : () -> ()
  ^bb2:  // pred: ^bb0
    "test.op"(%arg27) : (i1) -> ()
    "cf.br"()[^bb3] : () -> ()
  ^bb3:  // 2 preds: ^bb1, ^bb2
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32) -> (), sym_name = "switch_only_default"}> ({
  ^bb0(%arg25: i32, %arg26: f32):
    "test.termop"()[^bb1, ^bb2] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.switch"(%arg25, %arg26)[^bb2] <{case_operand_segments = array<i32>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (i32, f32) -> ()
  ^bb2(%23: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%23) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32, f32) -> (), sym_name = "switch_case_matching_default"}> ({
  ^bb0(%arg22: i32, %arg23: f32, %arg24: f32):
    "test.termop"()[^bb1, ^bb2, ^bb3] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.switch"(%arg22, %arg23, %arg23, %arg24, %arg23)[^bb2, ^bb2, ^bb3, ^bb2] <{case_operand_segments = array<i32: 1, 1, 1>, case_values = dense<[42, 10, 17]> : vector<3xi32>, operandSegmentSizes = array<i32: 1, 1, 3>}> : (i32, f32, f32, f32, f32) -> ()
  ^bb2(%21: f32):  // 4 preds: ^bb0, ^bb1, ^bb1, ^bb1
    "test.termop"(%21) : (f32) -> ()
  ^bb3(%22: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%22) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (f32, f32, f32) -> (), sym_name = "switch_on_const_no_match"}> ({
  ^bb0(%arg19: f32, %arg20: f32, %arg21: f32):
    "test.termop"()[^bb1, ^bb2, ^bb3, ^bb4] : () -> ()
  ^bb1:  // pred: ^bb0
    %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "cf.switch"(%17, %arg19, %arg20, %arg21)[^bb2, ^bb3, ^bb4] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[-1, 1]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, f32, f32, f32) -> ()
  ^bb2(%18: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%18) : (f32) -> ()
  ^bb3(%19: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%19) : (f32) -> ()
  ^bb4(%20: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%20) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (f32, f32, f32) -> (), sym_name = "switch_on_const_with_match"}> ({
  ^bb0(%arg16: f32, %arg17: f32, %arg18: f32):
    "test.termop"()[^bb1, ^bb2, ^bb3, ^bb4] : () -> ()
  ^bb1:  // pred: ^bb0
    %13 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "cf.switch"(%13, %arg16, %arg17, %arg18)[^bb2, ^bb3, ^bb4] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[-1, 1]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, f32, f32, f32) -> ()
  ^bb2(%14: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%14) : (f32) -> ()
  ^bb3(%15: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%15) : (f32) -> ()
  ^bb4(%16: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%16) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32, f32, f32, f32) -> (), sym_name = "switch_passthrough"}> ({
  ^bb0(%arg11: i32, %arg12: f32, %arg13: f32, %arg14: f32, %arg15: f32):
    "test.termop"()[^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.switch"(%arg11, %arg12, %arg13, %arg14)[^bb2, ^bb3, ^bb4] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[43, 44]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, f32, f32, f32) -> ()
  ^bb2(%8: f32):  // 2 preds: ^bb0, ^bb1
    "cf.br"(%8)[^bb5] : (f32) -> ()
  ^bb3(%9: f32):  // 2 preds: ^bb0, ^bb1
    "cf.br"(%9)[^bb6] : (f32) -> ()
  ^bb4(%10: f32):  // 2 preds: ^bb0, ^bb1
    "test.termop"(%10) : (f32) -> ()
  ^bb5(%11: f32):  // 2 preds: ^bb0, ^bb2
    "test.termop"(%11) : (f32) -> ()
  ^bb6(%12: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%12) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32, f32) -> (), sym_name = "switch_from_switch_with_same_value_with_match"}> ({
  ^bb0(%arg8: i32, %arg9: f32, %arg10: f32):
    "test.termop"()[^bb1, ^bb2, ^bb4, ^bb5] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.switch"(%arg8)[^bb2, ^bb3] <{case_operand_segments = array<i32: 0>, case_values = dense<42> : vector<1xi32>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "test.termop"() : () -> ()
  ^bb3:  // pred: ^bb1
    "test.op"() : () -> ()
    "cf.switch"(%arg8, %arg9, %arg10)[^bb4, ^bb5] <{case_operand_segments = array<i32: 1>, case_values = dense<42> : vector<1xi32>, operandSegmentSizes = array<i32: 1, 1, 1>}> : (i32, f32, f32) -> ()
  ^bb4(%6: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%6) : (f32) -> ()
  ^bb5(%7: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%7) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32, f32, f32) -> (), sym_name = "switch_from_switch_with_same_value_no_match"}> ({
  ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32):
    "test.termop"()[^bb1, ^bb2, ^bb4, ^bb5, ^bb6] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.switch"(%arg4)[^bb2, ^bb3] <{case_operand_segments = array<i32: 0>, case_values = dense<42> : vector<1xi32>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "test.termop"() : () -> ()
  ^bb3:  // pred: ^bb1
    "test.op"() : () -> ()
    "cf.switch"(%arg4, %arg5, %arg6, %arg7)[^bb4, ^bb5, ^bb6] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[0, 43]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, f32, f32, f32) -> ()
  ^bb4(%3: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%3) : (f32) -> ()
  ^bb5(%4: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%4) : (f32) -> ()
  ^bb6(%5: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%5) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, f32, f32, f32) -> (), sym_name = "switch_from_switch_default_with_same_value"}> ({
  ^bb0(%arg0: i32, %arg1: f32, %arg2: f32, %arg3: f32):
    "test.termop"()[^bb1, ^bb2, ^bb4, ^bb5, ^bb6] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.switch"(%arg0)[^bb3, ^bb2] <{case_operand_segments = array<i32: 0>, case_values = dense<42> : vector<1xi32>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    "test.termop"() : () -> ()
  ^bb3:  // pred: ^bb1
    "test.op"() : () -> ()
    "cf.switch"(%arg0, %arg1, %arg2, %arg3)[^bb4, ^bb5, ^bb6] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[42, 43]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, f32, f32, f32) -> ()
  ^bb4(%0: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%0) : (f32) -> ()
  ^bb5(%1: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%1) : (f32) -> ()
  ^bb6(%2: f32):  // 2 preds: ^bb0, ^bb3
    "test.termop"(%2) : (f32) -> ()
  }) : () -> ()
}) : () -> ()
