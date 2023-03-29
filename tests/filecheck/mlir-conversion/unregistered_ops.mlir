// RUN: xdsl-opt %s -t mlir --allow-unregistered-dialects | xdsl-opt -f mlir -t mlir --allow-unregistered-dialects  | filecheck %s

"builtin.module"() ({

  %0 = "region_op"() ({
    %y = "op_with_res"() {otherattr = #unknown_attr<b2 ...2 [] <<>>>} : () -> (i32)
    %z = "op_with_operands"(%y, %y) : (i32, i32) -> !unknown_type<{[<()>]}>
  }) {testattr = "foo"} : () -> i32

  // CHECK:       %{{.*}} = "region_op"() ({
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {"otherattr" = #unknown_attr<b2 ...2 [] <<>>>} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "op_with_operands"(%{{.*}}, %{{.*}}) : (i32, i32) -> !unknown_type<{[<()>]}>
  // CHECK-NEXT:  }) {"testattr" = "foo"} : () -> i32

}) : () -> ()