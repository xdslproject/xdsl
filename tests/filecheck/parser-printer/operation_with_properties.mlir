// RUN: xdsl-opt %s --allow-unregistered-dialect | filecheck %s

builtin.module {

    // An operation with a property
    "unregistered.op"() <{"test" = 2 : i32}> : () -> ()
    // CHECK: "unregistered.op"() <{"test" = 2 : i32}> : () -> ()

    // An operation with a property, a region, and an attribute
    "unregistered.op"() <{"test" = 42 : i64, "test2" = 71 : i32}> ({}) {"test3" = "foo"} : () -> i64
    // CHECK-NEXT: "unregistered.op"() <{"test" = 42 : i64, "test2" = 71 : i32}> ({
    // CHECK-NEXT: }) {"test" = "foo"} : () -> ()

    // An operation with a property and an attribute with the same name
    "unregistered.op"() <{"test" = 42 : i64}> ({}) {"test" = "foo"} : () -> i64
    // CHECK-NEXT: "unregistered.op"() <{"test" = 42 : i64}> {"test" = "foo"} : () -> ()

}