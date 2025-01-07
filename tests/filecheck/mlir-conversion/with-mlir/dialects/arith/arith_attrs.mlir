// RUN: mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({

// fastmath single flag set
"test.op"() {"fastmath" = #arith.fastmath<reassoc>}: ()->()
// CHECK:   fastmath = #arith.fastmath<reassoc>
"test.op"() {"fastmath" = #arith.fastmath<nnan>}: ()->()
// CHECK:   fastmath = #arith.fastmath<nnan>
"test.op"() {"fastmath" = #arith.fastmath<ninf>}: ()->()
// CHECK:   fastmath = #arith.fastmath<ninf>
"test.op"() {"fastmath" = #arith.fastmath<nsz>}: ()->()
// CHECK:   fastmath = #arith.fastmath<nsz>
"test.op"() {"fastmath" = #arith.fastmath<arcp>}: ()->()
// CHECK:   fastmath = #arith.fastmath<arcp>
"test.op"() {"fastmath" = #arith.fastmath<contract>}: ()->()
// CHECK:   fastmath = #arith.fastmath<contract>
"test.op"() {"fastmath" = #arith.fastmath<afn>}: ()->()
// CHECK:   fastmath = #arith.fastmath<afn>

// fastmath special cases
"test.op"() {"fastmath" = #arith.fastmath<none>}: ()->()
// CHECK:   fastmath = #arith.fastmath<none>
"test.op"() {"fastmath" = #arith.fastmath<fast>}: ()->()
// CHECK:   fastmath = #arith.fastmath<fast>

// fastmath combination
"test.op"() {"fastmath" = #arith.fastmath<nnan,nsz>}: ()->()
// CHECK:   fastmath = #arith.fastmath<nnan,nsz>

}) : ()->()
