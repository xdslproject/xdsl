// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | FileCheck %s

"builtin.module"() ({
  "func.func"() ({
    ^bb0(%arg0: index):
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "index_type"} : () -> ()

  // CHECK: ^{{.*}}(%{{.*}}: index):
}) : () -> ()