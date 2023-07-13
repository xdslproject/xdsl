// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

builtin.module {
    printf.print_format "Hello world!"

    %144 = "test.op"() : () -> i32
    %12 = "test.op"() : () -> i32

    printf.print_format "Uses vals twice {} {} {} {}", %12 : i32, %144 : i32, %12 : i32, %144 : i32

    printf.print_format "{}", %144 : i32
    printf.print_format "{}", %144 : i32 {unit}
}

// CHECK:       printf.print_format "Hello world!"
// CHECK-NEXT:  %0 = "test.op"() : () -> i32
// CHECK-NEXT:  %1 = "test.op"() : () -> i32
// CHECK-NEXT:  printf.print_format "Uses vals twice {} {} {} {}", %1 : i32, %0 : i32, %1 : i32, %0 : i32
// CHECK-NEXT:  printf.print_format "{}", %0 : i32
// CHECK-NEXT:  printf.print_format "{}", %0 : i32 {"unit"}
