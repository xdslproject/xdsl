// RUN: xdsl-opt --split-input-file --parsing-diagnostics %s | filecheck %s

%one = "test.op"() : () -> i32
%zero = arith.constant 0 : i32
%state = accfg.setup "acc1" to ("A" = %one : i32) : !accfg.state<"acc2">

// CHECK: expected !accfg.state<"acc1">, but got !accfg.state<"acc2">
