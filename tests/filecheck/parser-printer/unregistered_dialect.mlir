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
    %s = "op_with_res"() : () -> !unknowndialect.nested_brackets<[nested, <brackets>]>
    // Opaque syntax with slashes, nested brackets, strings
    "op"() {opaque_slash = #unknowndialect<myattr a / b>} : () -> ()
    "op"() {opaque_nested = #unknowndialect<myattr [x // y]>} : () -> ()
    "op"() {opaque_str = #unknowndialect<myattr "str>">} : () -> ()
    "op"() {escaped_str = #unknowndialect.esc<"a\"b">} : () -> ()
    %r = "op_with_res"() : () -> !unknowndialect<mytype a / b>
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
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() : () -> !unknowndialect.nested_brackets<[nested, <brackets>]>
  // CHECK-NEXT:   "op"() {opaque_slash = #unknowndialect<myattr a / b>} : () -> ()
  // CHECK-NEXT:   "op"() {opaque_nested = #unknowndialect<myattr [x // y]>} : () -> ()
  // CHECK-NEXT:   "op"() {opaque_str = #unknowndialect<myattr "str>">} : () -> ()
  // CHECK-NEXT:   "op"() {escaped_str = #unknowndialect.esc<"a\"b">} : () -> ()
  // CHECK-NEXT:   %{{.*}} = "op_with_res"() : () -> !unknowndialect<mytype a / b>
  // CHECK-NEXT:  }) {testattr = "foo"} : () -> i32
  // CHECK-NEXT:  "builtin.unimplemented_op"() {attr = #builtin.unimplemented_attr} : () -> ()

}) : () -> ()
