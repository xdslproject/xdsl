"builtin.module"() ({
  "memref.global"() <{initial_value = dense<[[1.000000e+00, -1.000000e+00, 0.000000e+00], [2.000000e+00, -2.000000e+00, 0.000000e+00]]> : tensor<2x3xf64>, sym_name = "a", sym_visibility = "public", type = memref<2x3xf64>}> : () -> ()
  "memref.global"() <{initial_value = dense<0.000000e+00> : tensor<2x3xf64>, sym_name = "b", sym_visibility = "public", type = memref<2x3xf64>}> : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "memref.get_global"() <{name = @a}> : () -> memref<2x3xf64>
    %1 = "memref.get_global"() <{name = @b}> : () -> memref<2x3xf64>
    %2 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
    "linalg.generic"(%0, %1) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg0: f64, %arg1: f64):
      %13 = "arith.maximumf"(%arg0, %2) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%13) : (f64) -> ()
    }) : (memref<2x3xf64>, memref<2x3xf64>) -> ()
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    %5 = "arith.constant"() <{value = 2 : index}> : () -> index
    %6 = "arith.constant"() <{value = 3 : index}> : () -> index
    %7 = "memref.load"(%1, %3, %3) : (memref<2x3xf64>, index, index) -> f64
    %8 = "memref.load"(%1, %3, %4) : (memref<2x3xf64>, index, index) -> f64
    %9 = "memref.load"(%1, %3, %5) : (memref<2x3xf64>, index, index) -> f64
    %10 = "memref.load"(%1, %4, %3) : (memref<2x3xf64>, index, index) -> f64
    %11 = "memref.load"(%1, %4, %4) : (memref<2x3xf64>, index, index) -> f64
    %12 = "memref.load"(%1, %4, %5) : (memref<2x3xf64>, index, index) -> f64
    "printf.print_format"(%7, %8, %9, %10, %11, %12) {format_str = "[[{}, {}, {}], [{}, {}, {}]]"} : (f64, f64, f64, f64, f64, f64) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
