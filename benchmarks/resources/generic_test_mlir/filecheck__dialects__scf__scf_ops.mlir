"builtin.module"() ({
  %0 = "test.op"() : () -> i1
  "scf.if"(%0) ({
    %38 = "test.op"() : () -> i32
    "scf.yield"() : () -> ()
  }, {
    %37 = "test.op"() : () -> i32
    "scf.yield"() : () -> ()
  }) : (i1) -> ()
  %1 = "scf.if"(%0) ({
    %36 = "test.op"() : () -> i32
    "scf.yield"(%36) : (i32) -> ()
  }, {
    %35 = "test.op"() : () -> i32
    "scf.yield"(%35) : (i32) -> ()
  }) : (i1) -> i32
  "scf.if"(%0) ({
    "scf.yield"() : () -> ()
  }, {
  }) : (i1) -> ()
  "func.func"() <{function_type = () -> (), sym_name = "while"}> ({
    %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %32 = "scf.while"(%31) ({
    ^bb0(%arg15: i32):
      %33 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %34 = "arith.cmpi"(%33, %arg15) <{predicate = 1 : i64}> : (i32, i32) -> i1
      "scf.condition"(%34, %33) : (i1, i32) -> ()
    }, {
    ^bb0(%arg14: i32):
      "scf.yield"(%arg14) : (i32) -> ()
    }) : (i32) -> i32
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "while2"}> ({
    %24 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %25 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %26:2 = "scf.while"(%25, %24) ({
    ^bb0(%arg12: i32, %arg13: f32):
      %29 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %30 = "arith.cmpi"(%arg12, %29) <{predicate = 0 : i64}> : (i32, i32) -> i1
      "scf.condition"(%30, %arg12, %arg13) : (i1, i32, f32) -> ()
    }, {
    ^bb0(%arg10: i32, %arg11: f32):
      %27 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %28 = "arith.addf"(%27, %arg11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "scf.yield"(%arg10, %28) : (i32, f32) -> ()
    }) : (i32, f32) -> (i32, f32)
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "while3"}> ({
    %17 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %18 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %19:2 = "scf.while"(%18, %17) ({
    ^bb0(%arg8: i32, %arg9: f32):
      %22 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %23 = "arith.cmpi"(%arg8, %22) <{predicate = 0 : i64}> : (i32, i32) -> i1
      "scf.condition"(%23, %arg8, %arg9) {hello = "world"} : (i1, i32, f32) -> ()
    }, {
    ^bb0(%arg6: i32, %arg7: f32):
      %20 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %21 = "arith.addf"(%20, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "scf.yield"(%arg6, %21) : (i32, f32) -> ()
    }) : (i32, f32) -> (i32, f32)
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "for"}> ({
    %11 = "arith.constant"() <{value = 0 : index}> : () -> index
    %12 = "arith.constant"() <{value = 42 : index}> : () -> index
    %13 = "arith.constant"() <{value = 3 : index}> : () -> index
    %14 = "arith.constant"() <{value = 1 : index}> : () -> index
    %15 = "scf.for"(%11, %12, %13, %14) ({
    ^bb0(%arg4: index, %arg5: index):
      %16 = "arith.muli"(%arg5, %arg4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "scf.yield"(%16) : (index) -> ()
    }) : (index, index, index, index) -> index
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "for_i32"}> ({
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %9 = "scf.for"(%5, %6, %7, %8) ({
    ^bb0(%arg2: i32, %arg3: i32):
      %10 = "arith.muli"(%arg3, %arg2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%10) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index) -> i32, sym_name = "index_switch"}> ({
  ^bb0(%arg1: index):
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %4:2 = "scf.index_switch"(%arg1) <{cases = array<i64: 1>}> ({
      "scf.yield"(%3, %3) : (i32, i32) -> ()
    }, {
      "scf.yield"(%2, %2) : (i32, i32) -> ()
    }) : (index) -> (i32, i32)
    "func.return"(%4#0) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index) -> (), sym_name = "switch_trivial"}> ({
  ^bb0(%arg0: index):
    "scf.index_switch"(%arg0) <{cases = array<i64>}> ({
      "scf.yield"() : () -> ()
    }) : (index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
