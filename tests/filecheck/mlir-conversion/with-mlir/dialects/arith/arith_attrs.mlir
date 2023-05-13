// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({

// fastmath single flag set
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_reassoc", "sym_visibility" = "private", "fastmath" = #arith.fastmath<reassoc>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<reassoc>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_nnan", "sym_visibility" = "private", "fastmath" = #arith.fastmath<nnan>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<nnan>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_ninf", "sym_visibility" = "private", "fastmath" = #arith.fastmath<ninf>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<ninf>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_nsz", "sym_visibility" = "private", "fastmath" = #arith.fastmath<nsz>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<nsz>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_arcp", "sym_visibility" = "private", "fastmath" = #arith.fastmath<arcp>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<arcp>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_contract", "sym_visibility" = "private", "fastmath" = #arith.fastmath<contract>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<contract>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_afn", "sym_visibility" = "private", "fastmath" = #arith.fastmath<afn>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<afn>

// fastmath special cases
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_none", "sym_visibility" = "private", "fastmath" = #arith.fastmath<none>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<none>
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_fast", "sym_visibility" = "private", "fastmath" = #arith.fastmath<fast>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<fast>

// fastmath combination
"func.func"() ({}) {"function_type" = ()->(), "sym_name" = "fast_two", "sym_visibility" = "private", "fastmath" = #arith.fastmath<nnan,nsz>}: ()->()
// CHECK:   "fastmath" = #arith.fastmath<nnan,nsz>

}) : ()->()
