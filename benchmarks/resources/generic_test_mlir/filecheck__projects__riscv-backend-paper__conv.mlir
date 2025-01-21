"builtin.module"() ({
  "memref.global"() <{initial_value = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00, 9.000000e+00]]]]> : tensor<1x1x3x3xf64>, sym_name = "a", sym_visibility = "public", type = memref<1x1x3x3xf64>}> : () -> ()
  "memref.global"() <{initial_value = dense<[[[[0.000000e+00, 2.500000e-01], [5.000000e-01, 7.500000e-01]]]]> : tensor<1x1x2x2xf64>, sym_name = "b", sym_visibility = "public", type = memref<1x1x2x2xf64>}> : () -> ()
  "memref.global"() <{initial_value = dense<0.000000e+00> : tensor<1x1x2x2xf64>, sym_name = "c", sym_visibility = "public", type = memref<1x1x2x2xf64>}> : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main", sym_visibility = "public"}> ({
    %0 = "memref.get_global"() <{name = @a}> : () -> memref<1x1x3x3xf64>
    %1 = "memref.get_global"() <{name = @b}> : () -> memref<1x1x2x2xf64>
    %2 = "memref.get_global"() <{name = @c}> : () -> memref<1x1x2x2xf64>
    "linalg.generic"(%0, %1, %2) <{indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %9 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %10 = "arith.addf"(%arg2, %9) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%10) : (f64) -> ()
    }) : (memref<1x1x3x3xf64>, memref<1x1x2x2xf64>, memref<1x1x2x2xf64>) -> ()
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    %5 = "memref.load"(%2, %3, %3, %3, %3) : (memref<1x1x2x2xf64>, index, index, index, index) -> f64
    %6 = "memref.load"(%2, %3, %3, %3, %4) : (memref<1x1x2x2xf64>, index, index, index, index) -> f64
    %7 = "memref.load"(%2, %3, %3, %4, %3) : (memref<1x1x2x2xf64>, index, index, index, index) -> f64
    %8 = "memref.load"(%2, %3, %3, %4, %4) : (memref<1x1x2x2xf64>, index, index, index, index) -> f64
    "printf.print_format"(%5, %6, %7, %8) {format_str = "[[[[{}, {}], [{}, {}]]]]"} : (f64, f64, f64, f64) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
