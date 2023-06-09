// RUN: xdsl-opt %s | filecheck %s

builtin.module {
    print.println "Hello world!"

    %144 = "arith.constant"() {"value" = 144 : i32} : () -> i32
    %12 = "arith.constant"() {"value" = 12 : i32} : () -> i32

    print.println "Did you know that {}^2 = {}, and fib({}) = {} too!", %12 : i32, %144 : i32, %12 : i32, %144 : i32

    print.println "Furthermore, {} is the smallest number with 15 divisors", %144 : i32 {"test_thing" = 32 : i32}
}

// CHECK:      builtin.module {
// CHECK-NEXT:   print.println "Hello world!"
// CHECK-NEXT:   %0 = "arith.constant"() {"value" = 144 : i32} : () -> i32
// CHECK-NEXT:   %1 = "arith.constant"() {"value" = 12 : i32} : () -> i32
// CHECK-NEXT:   print.println "Did you know that {}^2 = {}, and fib({}) = {} too!", %1 : i32, %0 : i32, %1 : i32, %0 : i32
// CHECK-NEXT:   print.println "Furthermore, {} is the smallest number with 15 divisors", %0 : i32 {test_thing=32 : i32}
// CHECK-NEXT: }
