"builtin.module"() ({
  "csl_wrapper.module"() <{height = 10 : i16, params = [#csl_wrapper.param<"z_dim" default=4: i16>, #csl_wrapper.param<"pattern" : i16>], width = 10 : i16}> ({
  ^bb0(%arg7: i16, %arg8: i16, %arg9: i16, %arg10: i16, %arg11: i16, %arg12: i16):
    %0 = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %1 = "csl.get_color"(%0) : (i16) -> !csl.color
    %2 = "csl_wrapper.import"(%arg12, %arg9, %arg10) <{fields = ["pattern", "peWidth", "peHeight"], module = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
    %3 = "csl_wrapper.import"(%arg9, %arg10, %1) <{fields = ["width", "height", "LAUNCH"], module = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
    %4 = "csl.member_call"(%2, %arg7, %arg8, %arg10, %arg9, %arg12) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %5 = "csl.member_call"(%3, %arg7) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %6 = "arith.constant"() <{value = 1 : i16}> : () -> i16
    %7 = "arith.minsi"(%arg12, %6) : (i16, i16) -> i16
    %8 = "arith.minsi"(%arg9, %arg7) : (i16, i16) -> i16
    %9 = "arith.minsi"(%arg10, %arg8) : (i16, i16) -> i16
    %10 = "arith.cmpi"(%arg7, %7) <{predicate = 2 : i64}> : (i16, i16) -> i1
    %11 = "arith.cmpi"(%arg8, %7) <{predicate = 2 : i64}> : (i16, i16) -> i1
    %12 = "arith.cmpi"(%8, %arg12) <{predicate = 2 : i64}> : (i16, i16) -> i1
    %13 = "arith.cmpi"(%9, %arg12) <{predicate = 2 : i64}> : (i16, i16) -> i1
    %14 = "arith.ori"(%10, %11) : (i1, i1) -> i1
    %15 = "arith.ori"(%14, %12) : (i1, i1) -> i1
    %16 = "arith.ori"(%15, %13) : (i1, i1) -> i1
    "csl_wrapper.yield"(%5, %4, %16) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^bb0(%arg0: i16, %arg1: i16, %arg2: i16, %arg3: i16, %arg4: !csl.comptime_struct, %arg5: !csl.comptime_struct, %arg6: i1):
    "func.func"() <{function_type = () -> (), sym_name = "gauss_seidel"}> ({
      "func.return"() : () -> ()
    }) : () -> ()
    "csl_wrapper.yield"() <{fields = []}> : () -> ()
  }) : () -> ()
}) : () -> ()
