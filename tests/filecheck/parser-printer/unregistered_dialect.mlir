// RUN: xdsl-opt %s --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect  | filecheck %s

"builtin.module"() ({

  %0 = "region_op"() ({
    %y = "op_with_res"() {otherattr = #unknown_attr<b2 ...2 [] <<>>>} : () -> (i32)
    %z = "op_with_operands"(%y, %y) : (i32, i32) -> !unknown_type<{[<()>]}>
    "op"() {ab = !unknown_singleton_type} : () -> ()
  }) {testattr = "foo"} : () -> i32

  // CHECK:       %{{.*}} = "region_op"() ({
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {"otherattr" = #unknown_attr<b2 ...2 [] <<>>>} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "op_with_operands"(%{{.*}}, %{{.*}}) : (i32, i32) -> !unknown_type<{[<()>]}>
  // CHECK-NEXT:   "op"() {"ab" = !unknown_singleton_type} : () -> ()
  // CHECK-NEXT:  }) {"testattr" = "foo"} : () -> i32

}) : () -> ()
