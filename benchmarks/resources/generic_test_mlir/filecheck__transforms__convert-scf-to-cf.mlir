"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "triangle"}> ({
  ^bb0(%arg6: i32):
    %16 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %19 = "scf.for"(%17, %arg6, %18, %16) ({
    ^bb0(%arg7: i32, %arg8: i32):
      %20 = "arith.addi"(%arg8, %arg7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%20) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32
    "func.return"(%19) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1) -> i32, sym_name = "if"}> ({
  ^bb0(%arg5: i1):
    %13 = "scf.if"(%arg5) ({
      %15 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.yield"(%15) : (i32) -> ()
    }, {
      %14 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      "scf.yield"(%14) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%13) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i1) -> (), sym_name = "if_no_else"}> ({
  ^bb0(%arg4: i1):
    "scf.if"(%arg4) ({
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index) -> index, sym_name = "nested"}> ({
  ^bb0(%arg1: index):
    %5 = "arith.constant"() <{value = 0 : index}> : () -> index
    %6 = "arith.constant"() <{value = 0 : index}> : () -> index
    %7 = "arith.constant"() <{value = 1 : index}> : () -> index
    %8 = "arith.constant"() <{value = 2 : index}> : () -> index
    %9 = "scf.for"(%6, %arg1, %7, %5) ({
    ^bb0(%arg2: index, %arg3: index):
      %10 = "arith.constant"() <{value = true}> : () -> i1
      %11 = "scf.if"(%10) ({
        %12 = "arith.addi"(%arg3, %arg2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.yield"(%12) : (index) -> ()
      }, {
        "scf.yield"(%arg3) : (index) -> ()
      }) : (i1) -> index
      "scf.yield"(%11) : (index) -> ()
    }) : (index, index, index, index) -> index
    "func.return"(%9) : (index) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index) -> i32, sym_name = "index_switch"}> ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2:2 = "scf.index_switch"(%arg0) <{cases = array<i64: 0, 1>}> ({
      "scf.yield"(%1, %1) : (i32, i32) -> ()
    }, {
      %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %4 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.yield"(%3, %4) : (i32, i32) -> ()
    }, {
      "scf.yield"(%0, %0) : (i32, i32) -> ()
    }) : (index) -> (i32, i32)
    "func.return"(%2#0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
