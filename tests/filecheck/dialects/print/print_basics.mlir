// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

builtin.module {
    print.println "Hello world!"

    %144 = "test.op"() : () -> i32
    %12 = "test.op"() : () -> i32

    print.println "Uses vals twice {} {} {} {}", %12 : i32, %144 : i32, %12 : i32, %144 : i32

    print.println "{}", %144 : i32
    print.println "{}", %144 : i32 {unit}
}

// CHECK:       print.println "Hello world!"
// CHECK-NEXT:  %0 = "test.op"() : () -> i32
// CHECK-NEXT:  %1 = "test.op"() : () -> i32
// CHECK-NEXT:  print.println "Uses vals twice {} {} {} {}", %1 : i32, %0 : i32, %1 : i32, %0 : i32
// CHECK-NEXT:  print.println "{}", %0 : i32
// CHECK-NEXT:  print.println "{}", %0 : i32 {"unit"}
