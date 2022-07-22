module ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.ceildivsi"(%arg0, %arg1) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "ceildivi"} : () -> ()
  "func.func"() ({
  ^bb2(%arg2: index, %arg3: index):
    %2 = "arith.ceildivsi"(%arg2, %arg3) : (index, index) -> index
    "func.return"(%2) : (index) -> ()
  }) {function_type = (index, index) -> index, sym_name = "ceildivi_index"} : () -> ()
  "func.func"() ({
  ^bb4(%arg4: i32, %arg5: i32):
    %4 = "arith.floordivsi"(%arg4, %arg5) : (i32, i32) -> i32
    "func.return"(%4) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "floordivi"} : () -> ()
  "func.func"() ({
  ^bb6(%arg6: index, %arg7: index):
    %6 = "arith.floordivsi"(%arg6, %arg7) : (index, index) -> index
    "func.return"(%6) : (index) -> ()
  }) {function_type = (index, index) -> index, sym_name = "floordivi_index"} : () -> ()
  "func.func"() ({
  ^bb8(%arg8: i32, %arg9: i32):
    %8 = "arith.ceildivui"(%arg8, %arg9) : (i32, i32) -> i32
    "func.return"(%8) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "ceildivui"} : () -> ()
  "func.func"() ({
  ^bb10(%arg10: index, %arg11: index):
    %10 = "arith.ceildivui"(%arg10, %arg11) : (index, index) -> index
    "func.return"(%10) : (index) -> ()
  }) {function_type = (index, index) -> index, sym_name = "ceildivui_index"} : () -> ()
  "func.func"() ({
  ^bb12(%arg12: f32, %arg13: f32):
    %12 = "arith.maxf"(%arg12, %arg13) : (f32, f32) -> f32
    "func.return"(%12) : (f32) -> ()
  }) {function_type = (f32, f32) -> f32, sym_name = "maxf"} : () -> ()
  "func.func"() ({
  ^bb14(%arg14: vector<4xf16>, %arg15: vector<4xf16>):
    %14 = "arith.maxf"(%arg14, %arg15) : (vector<4xf16>, vector<4xf16>) -> vector<4xf16>
    "func.return"(%14) : (vector<4xf16>) -> ()
  }) {function_type = (vector<4xf16>, vector<4xf16>) -> vector<4xf16>, sym_name = "maxf_vector"} : () -> ()
  "func.func"() ({
  ^bb16(%arg16: f32, %arg17: f32):
    %16 = "arith.minf"(%arg16, %arg17) : (f32, f32) -> f32
    "func.return"(%16) : (f32) -> ()
  }) {function_type = (f32, f32) -> f32, sym_name = "minf"} : () -> ()
  "func.func"() ({
  ^bb18(%arg1+8: i32, %arg19: i32):
    %18 = "arith.maxsi"(%arg18, %arg19) : (i32, i32) -> i32
    "func.return"(%18) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "maxsi"} : () -> ()
  "func.func"() ({
  ^bb20(%arg20: i32, %arg21: i32):
    %20 = "arith.minsi"(%arg20, %arg21) : (i32, i32) -> i32
    "func.return"(%20) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "minsi"} : () -> ()
  "func.func"() ({
  ^bb22(%arg22: i32, %arg23: i32):
    %22 = "arith.maxui"(%arg22, %arg23) : (i32, i32) -> i32
    "func.return"(%22) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "maxui"} : () -> ()
  "func.func"() ({
  ^bb24(%arg24: i32, %arg25: i32):
    %24 = "arith.minui"(%arg24, %arg25) : (i32, i32) -> i32
    "func.return"(%24) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "minui"} : () -> ()
}) : () -> ()

