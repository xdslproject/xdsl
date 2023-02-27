// RUN: xdsl-opt --parsing-diagnostics %s | filecheck %s

"builtin.module"() ({
}) {"wrong_all_reduce_operation" = #gpu<all_reduce_op magic>}: () -> ()

// CHECK: Unexpected op magic. A gpu all_reduce_op can only be add, and, max, min, mul, or, or xor
