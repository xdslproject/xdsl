// RUN: xdsl-opt %s --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect  | filecheck %s

"builtin.module"() ({

  %0 = "region_op"() ({
    %x = "op_with_res"() {otherattr = #unknowndialect.unknown_attr<b2 ...2 [] <<>>>} : () -> (i32)
    %y = "op_with_res"() {otherattr = #unknowndialect<b2 ...2 [] <<>>>} : () -> (i32)
    %z = "op_with_operands"(%y, %y) : (i32, i32) -> !unknowndialect.unknown_type<{[<()>]}>
    "op"() {ab = !unknowndialect.unknown_singleton_type} : () -> ()
    %w = "op_with_res"() : () -> !unknowndialect.type_with_slash<a / b>
    %v = "op_with_res"() : () -> !unknowndialect.type_with_slashes<valid / mlir // syntax ///>
    %u = "op_with_res"() : () -> !unknowndialect.type_with_nested_slash<a, [b // c], d>
    %t = "op_with_res"() : () -> !unknowndialect.type_with_slashes<a // b> // comment after attribute
    "op"() {slash_attr = #unknowndialect.attr_with_slash<x // y>} : () -> ()
    "op"() {arrow_attr = #unknowndialect.attr_with_arrow<a -> b>} : () -> ()
    "op"() {str_attr = #unknowndialect.attr_with_str<"hello // world">} : () -> ()
  }) {testattr = "foo"} : () -> i32
  "builtin.unimplemented_op"() {"attr" = #builtin.unimplemented_attr} : () -> ()


  // CHECK:       %{{.*}} = "region_op"() ({
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {otherattr = #unknowndialect.unknown_attr<b2 ...2 [] <<>>>} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() {otherattr = #unknowndialect<b2 ...2 [] <<>>>} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "op_with_operands"(%{{.*}}, %{{.*}}) : (i32, i32) -> !unknowndialect.unknown_type<{[<()>]}>
  // CHECK-NEXT:   "op"() {ab = !unknowndialect.unknown_singleton_type} : () -> ()
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() : () -> !unknowndialect.type_with_slash<a / b>
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() : () -> !unknowndialect.type_with_slashes<valid / mlir // syntax ///>
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() : () -> !unknowndialect.type_with_nested_slash<a, [b // c], d>
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() : () -> !unknowndialect.type_with_slashes<a // b>
  // CHECK-NEXT:   "op"() {slash_attr = #unknowndialect.attr_with_slash<x // y>} : () -> ()
  // CHECK-NEXT:   "op"() {arrow_attr = #unknowndialect.attr_with_arrow<a -> b>} : () -> ()
  // CHECK-NEXT:   "op"() {str_attr = #unknowndialect.attr_with_str<"hello // world">} : () -> ()
  // CHECK-NEXT:  }) {testattr = "foo"} : () -> i32
  // CHECK-NEXT:  "builtin.unimplemented_op"() {attr = #builtin.unimplemented_attr} : () -> ()

}) : () -> ()
