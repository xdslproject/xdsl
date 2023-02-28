// RUN: xdsl-opt --parsing-diagnostics %s | filecheck %s

"builtin.module"() ({
}) {"wrong_dim" = #gpu<dim w>}: () -> ()

// CHECK: Unexpected dim w. A gpu dim can only be x, y, or z
