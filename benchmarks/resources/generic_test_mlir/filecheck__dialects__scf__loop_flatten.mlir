"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 1 : index}> : () -> index
  %2 = "arith.constant"() <{value = 3 : index}> : () -> index
  %3 = "arith.constant"() <{value = 5 : index}> : () -> index
  %4 = "arith.constant"() <{value = 8 : index}> : () -> index
  %5 = "arith.constant"() <{value = 64 : index}> : () -> index
  %6 = "test.op"() : () -> index
  %7:3 = "test.op"() : () -> (index, index, f32)
  "scf.for"(%6, %5, %4) ({
  ^bb0(%arg85: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg86: index):
      %65 = "arith.constant"() <{value = 8 : index}> : () -> index
      %66 = "arith.addi"(%arg85, %arg86) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%66) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %3) ({
  ^bb0(%arg83: index):
    "scf.for"(%0, %4, %2) ({
    ^bb0(%arg84: index):
      %64 = "arith.constant"() <{value = 8 : index}> : () -> index
      "test.op"(%64) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %8:3 = "scf.for"(%0, %5, %4, %7#1, %7#1, %7#2) ({
  ^bb0(%arg75: index, %arg76: index, %arg77: index, %arg78: f32):
    %61:3 = "scf.for"(%0, %4, %1, %arg76, %arg77, %arg78) ({
    ^bb0(%arg79: index, %arg80: index, %arg81: index, %arg82: f32):
      %62 = "arith.constant"() <{value = 8 : index}> : () -> index
      %63 = "arith.addi"(%arg75, %arg79) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%63) : (index) -> ()
      "scf.yield"(%arg80, %arg81, %arg82) : (index, index, f32) -> ()
    }) : (index, index, index, index, index, f32) -> (index, index, f32)
    "scf.yield"(%61#0, %61#1, %61#2) : (index, index, f32) -> ()
  }) : (index, index, index, index, index, f32) -> (index, index, f32)
  %9:3 = "scf.for"(%0, %5, %4, %7#1, %7#1, %7#2) ({
  ^bb0(%arg67: index, %arg68: index, %arg69: index, %arg70: f32):
    %59:3 = "scf.for"(%0, %4, %1, %arg68, %arg69, %arg70) ({
    ^bb0(%arg71: index, %arg72: index, %arg73: index, %arg74: f32):
      %60 = "arith.constant"() <{value = 8 : index}> : () -> index
      "test.op"(%60) : (index) -> ()
      "scf.yield"(%arg72, %arg73, %arg74) : (index, index, f32) -> ()
    }) : (index, index, index, index, index, f32) -> (index, index, f32)
    "scf.yield"(%59#0, %59#1, %59#2) : (index, index, f32) -> ()
  }) : (index, index, index, index, index, f32) -> (index, index, f32)
  %10:3 = "scf.for"(%0, %5, %4, %7#1, %7#1, %7#2) ({
  ^bb0(%arg59: index, %arg60: index, %arg61: index, %arg62: f32):
    %56:3 = "scf.for"(%0, %4, %1, %arg60, %arg61, %arg62) ({
    ^bb0(%arg63: index, %arg64: index, %arg65: index, %arg66: f32):
      %57 = "arith.constant"() <{value = 8 : index}> : () -> index
      %58 = "test.op"(%57) : (index) -> index
      "scf.yield"(%58, %arg65, %arg66) : (index, index, f32) -> ()
    }) : (index, index, index, index, index, f32) -> (index, index, f32)
    "scf.yield"(%56#0, %56#1, %56#2) : (index, index, f32) -> ()
  }) : (index, index, index, index, index, f32) -> (index, index, f32)
  %11 = "scf.for"(%0, %5, %4, %0) ({
  ^bb0(%arg56: index, %arg57: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg58: index):
      %54 = "arith.constant"() <{value = 8 : index}> : () -> index
      %55 = "arith.addi"(%arg56, %arg58) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%55) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"(%arg57) : (index) -> ()
  }) : (index, index, index, index) -> index
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg54: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg55: index):
      %52 = "arith.constant"() <{value = 8 : index}> : () -> index
      %53 = "arith.addi"(%arg54, %arg55) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%53) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %51 = "arith.constant"() <{value = 42 : index}> : () -> index
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg52: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg53: index):
      "test.op"(%arg52) : (index) -> ()
      "test.op"(%arg53) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg49: index):
    %48 = "scf.for"(%0, %4, %1, %0) ({
    ^bb0(%arg50: index, %arg51: index):
      %49 = "arith.constant"() <{value = 8 : index}> : () -> index
      %50 = "arith.addi"(%arg49, %arg50) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%50) : (index) -> ()
      "scf.yield"(%arg51) : (index) -> ()
    }) : (index, index, index, index) -> index
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg47: index):
    "scf.for"(%4, %4, %1) ({
    ^bb0(%arg48: index):
      %46 = "arith.constant"() <{value = 8 : index}> : () -> index
      %47 = "arith.addi"(%arg47, %arg48) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%47) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg45: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg46: index):
      %44 = "arith.constant"() <{value = 8 : index}> : () -> index
      %45 = "arith.addi"(%arg45, %arg46) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%45, %arg45) : (index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg43: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg44: index):
      %42 = "arith.constant"() <{value = 8 : index}> : () -> index
      %43 = "arith.addi"(%arg43, %arg44) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%43, %arg44) : (index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg41: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg42: index):
      %40 = "arith.constant"() <{value = 8 : index}> : () -> index
      %41 = "arith.muli"(%arg41, %arg42) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%41) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%6, %5, %4) ({
  ^bb0(%arg39: index):
    "scf.for"(%0, %4, %6) ({
    ^bb0(%arg40: index):
      %38 = "arith.constant"() <{value = 8 : index}> : () -> index
      %39 = "arith.addi"(%arg39, %arg40) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%39) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%6, %5, %4) ({
  ^bb0(%arg37: index):
    "scf.for"(%6, %4, %1) ({
    ^bb0(%arg38: index):
      %36 = "arith.constant"() <{value = 8 : index}> : () -> index
      %37 = "arith.addi"(%arg37, %arg38) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%37) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%6, %5, %4) ({
  ^bb0(%arg35: index):
    "scf.for"(%0, %6, %1) ({
    ^bb0(%arg36: index):
      %34 = "arith.constant"() <{value = 8 : index}> : () -> index
      %35 = "arith.addi"(%arg35, %arg36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%35) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%6, %5, %6) ({
  ^bb0(%arg33: index):
    "scf.for"(%0, %4, %1) ({
    ^bb0(%arg34: index):
      %32 = "arith.constant"() <{value = 8 : index}> : () -> index
      %33 = "arith.addi"(%arg33, %arg34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%33) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg31: index):
    "scf.for"(%0, %4, %2) ({
    ^bb0(%arg32: index):
      %30 = "arith.constant"() <{value = 8 : index}> : () -> index
      %31 = "arith.addi"(%arg31, %arg32) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%31) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg29: index):
    "scf.for"(%0, %4, %2) ({
    ^bb0(%arg30: index):
      %28 = "arith.constant"() <{value = 8 : index}> : () -> index
      %29 = "arith.addi"(%arg29, %arg30) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%29) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%0, %5, %4) ({
  ^bb0(%arg27: index):
    "scf.for"(%0, %3, %2) ({
    ^bb0(%arg28: index):
      %26 = "arith.constant"() <{value = 8 : index}> : () -> index
      %27 = "arith.addi"(%arg27, %arg28) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%27) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%1, %5, %3) ({
  ^bb0(%arg25: index):
    "scf.for"(%0, %4, %2) ({
    ^bb0(%arg26: index):
      %25 = "arith.constant"() <{value = 8 : index}> : () -> index
      "test.op"(%25) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%6, %5, %3) ({
  ^bb0(%arg23: index):
    "scf.for"(%0, %4, %2) ({
    ^bb0(%arg24: index):
      %24 = "arith.constant"() <{value = 8 : index}> : () -> index
      "test.op"(%24) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %12:3 = "scf.for"(%0, %5, %4, %7#1, %7#1, %7#2) ({
  ^bb0(%arg15: index, %arg16: index, %arg17: index, %arg18: f32):
    %21:3 = "scf.for"(%0, %4, %1, %arg17, %arg16, %arg18) ({
    ^bb0(%arg19: index, %arg20: index, %arg21: index, %arg22: f32):
      %22 = "arith.constant"() <{value = 8 : index}> : () -> index
      %23 = "arith.addi"(%arg15, %arg19) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%23) : (index) -> ()
      "scf.yield"(%arg20, %arg21, %arg22) : (index, index, f32) -> ()
    }) : (index, index, index, index, index, f32) -> (index, index, f32)
    "scf.yield"(%21#0, %21#1, %21#2) : (index, index, f32) -> ()
  }) : (index, index, index, index, index, f32) -> (index, index, f32)
  %13:3 = "scf.for"(%0, %5, %4, %7#1, %7#1, %7#2) ({
  ^bb0(%arg8: index, %arg9: index, %arg10: index, %arg11: f32):
    %18:2 = "scf.for"(%0, %4, %1, %arg9, %arg10) ({
    ^bb0(%arg12: index, %arg13: index, %arg14: index):
      %19 = "arith.constant"() <{value = 8 : index}> : () -> index
      %20 = "arith.addi"(%arg8, %arg12) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%20) : (index) -> ()
      "scf.yield"(%arg13, %arg14) : (index, index) -> ()
    }) : (index, index, index, index, index) -> (index, index)
    "scf.yield"(%18#0, %18#1, %arg11) : (index, index, f32) -> ()
  }) : (index, index, index, index, index, f32) -> (index, index, f32)
  %14:3 = "scf.for"(%0, %5, %4, %7#1, %7#1, %7#2) ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: f32):
    %15:3 = "scf.for"(%0, %4, %1, %arg1, %arg2, %arg3) ({
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: f32):
      %16 = "arith.constant"() <{value = 8 : index}> : () -> index
      %17 = "arith.addi"(%arg0, %arg4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "test.op"(%17) : (index) -> ()
      "scf.yield"(%arg5, %arg6, %arg7) : (index, index, f32) -> ()
    }) : (index, index, index, index, index, f32) -> (index, index, f32)
    "scf.yield"(%15#1, %15#0, %15#2) : (index, index, f32) -> ()
  }) : (index, index, index, index, index, f32) -> (index, index, f32)
}) : () -> ()
