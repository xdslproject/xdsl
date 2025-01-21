"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "assert", sym_visibility = "private"}> ({
    %11 = "arith.constant"() <{value = true}> : () -> i1
    "cf.assert"(%11) <{msg = "some message"}> : (i1) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "unconditional_br", sym_visibility = "private"}> ({
    "cf.br"()[^bb1] : () -> ()
  ^bb1:  // 2 preds: ^bb0, ^bb1
    "cf.br"()[^bb1] : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> (), sym_name = "br", sym_visibility = "private"}> ({
  ^bb0(%arg3: i32):
    "cf.br"(%arg3)[^bb1] : (i32) -> ()
  ^bb1(%10: i32):  // 2 preds: ^bb0, ^bb1
    "cf.br"(%10)[^bb1] : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1, i32) -> i32, sym_name = "cond_br", sym_visibility = "private"}> ({
  ^bb0(%arg1: i1, %arg2: i32):
    "cf.br"(%arg1, %arg2)[^bb1] : (i1, i32) -> ()
  ^bb1(%5: i1, %6: i32):  // 2 preds: ^bb0, ^bb1
    "cf.cond_br"(%5, %5, %6, %6, %6, %6)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 2, 3>}> : (i1, i1, i32, i32, i32, i32) -> ()
  ^bb2(%7: i32, %8: i32, %9: i32):  // pred: ^bb1
    "func.return"(%7) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> (), sym_name = "switch"}> ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "cf.switch"(%arg0, %0, %1, %1)[^bb1, ^bb2, ^bb3] <{case_operand_segments = array<i32: 2, 0>, case_values = dense<[42, 43]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, i32, i32, i32) -> ()
  ^bb1(%2: i32):  // pred: ^bb0
    "func.return"() : () -> ()
  ^bb2(%3: i32, %4: i32):  // pred: ^bb0
    "func.return"() : () -> ()
  ^bb3:  // pred: ^bb0
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
