// RUN: XDSL_AUTO_ROUNDTRIP
builtin.module {
    printf.print_format "Hello world!"

    %144 = "test.op"() : () -> i32
    %12 = "test.op"() : () -> i32
    %byte = "test.op"() : () -> i8

    printf.print_format "Uses vals twice {} {} {} {}", %12 : i32, %144 : i32, %12 : i32, %144 : i32
    printf.print_format "{}", %144 : i32
    printf.print_format "{}", %144 : i32 {"unit"}
    "printf.print_char"(%byte) : (i8) -> ()
    "printf.print_int"(%12) : (i32) -> ()
}
