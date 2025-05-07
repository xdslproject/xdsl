// RUN: xdsl-opt %s --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect  | filecheck %s

"builtin.module"() ({

  %0 = "region_op"() ({
    %x = "op_with_res"() {otherattr = #unknowndialect.unknown_attr<b2 ...2 [] <<>>>} : () -> (i32)
    %y = "op_with_res"() {otherattr = #unknowndialect<b2 ...2 [] <<>>>} : () -> (i32)
    %z = "op_with_operands"(%y, %y) : (i32, i32) -> !unknowndialect.unknown_type<{[<()>]}>
    "op"() {ab = !unknowndialect.unknown_singleton_type} : () -> ()
  }) {testattr = "foo"} : () -> i32
  "builtin.unimplemented_op"() {"attr" = #builtin.unimplemented_attr} : () -> ()


  // CHECK:       %{{.*}} = "region_op"() ({
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {otherattr = #unknowndialect.unknown_attr<b2 ...2 [] <<>>>} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {otherattr = #unknowndialect<b2 ...2 [] <<>>>} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "op_with_operands"(%{{.*}}, %{{.*}}) : (i32, i32) -> !unknowndialect.unknown_type<{[<()>]}>
  // CHECK-NEXT:   "op"() {ab = !unknowndialect.unknown_singleton_type} : () -> ()
  // CHECK-NEXT:  }) {testattr = "foo"} : () -> i32
  // CHECK-NEXT:  "builtin.unimplemented_op"() {attr = #builtin.unimplemented_attr} : () -> ()

}) : () -> ()
