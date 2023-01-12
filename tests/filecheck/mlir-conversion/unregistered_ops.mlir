// RUN: xdsl-opt %s -t mlir --allow-unregistered-ops | xdsl-opt -f mlir -t mlir --allow-unregistered-ops  | filecheck %s

"builtin.module"() ({

  %0 = "region_op"() ({
    %y = "op_with_res"() {otherattr = 3 : i32} : () -> (i32)
    "op_with_operands"(%y, %y) : (i32, i32) -> ()
  }) {testattr = "foo"} : () -> i32

  // CHECK:       %{{.*}} = "region_op"() ({
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {"otherattr" = 3 : i32} : () -> i32
  // CHECK-NEXT:   "op_with_operands"(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
  // CHECK-NEXT:  }) {"testattr" = "foo"} : () -> i32

}) : () -> ()