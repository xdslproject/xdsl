// RUN: XDSL_ROUNDTRIP
builtin.module {
    printf.print_format "Hello world!\n"

    %144 = "test.op"() : () -> i32
    %12 = "test.op"() : () -> i32
    %byte = "test.op"() : () -> i8

    printf.print_format "Uses vals twice {} {} {} {}\n", %12 : i32, %144 : i32, %12 : i32, %144 : i32
    printf.print_format "{}\n", %144 : i32
    printf.print_format "{}\n", %144 : i32 {unit}
    "printf.print_char"(%byte) : (i8) -> ()
    "printf.print_int"(%12) : (i32) -> ()
}

// CHECK:       printf.print_format "Hello world!\n"
// CHECK-NEXT:  %0 = "test.op"() : () -> i32
// CHECK-NEXT:  %1 = "test.op"() : () -> i32
// CHECK-NEXT:  %byte = "test.op"() : () -> i8
// CHECK-NEXT:  printf.print_format "Uses vals twice {} {} {} {}\n", %1 : i32, %0 : i32, %1 : i32, %0 : i32
// CHECK-NEXT:  printf.print_format "{}\n", %0 : i32
// CHECK-NEXT:  printf.print_format "{}\n", %0 : i32 {unit}
// CHECK-NEXT:  "printf.print_char"(%{{.*}}) : (i8) -> ()
// CHECK-NEXT:  "printf.print_int"(%{{.*}}) : (i32) -> ()
