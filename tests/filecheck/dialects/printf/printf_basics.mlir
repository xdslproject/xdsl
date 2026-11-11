// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       printf.print_format "Hello world!\0A"
printf.print_format "Hello world!\0A"

%144 = "test.op"() : () -> i32
%12 = "test.op"() : () -> i32
%byte = "test.op"() : () -> i8

// CHECK:  printf.print_format "Uses vals twice {} {} {} {}\0A", %{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32
printf.print_format "Uses vals twice {} {} {} {}\0A", %12 : i32, %144 : i32, %12 : i32, %144 : i32
// CHECK-NEXT:  printf.print_format "{}\0A", %{{.*}} : i32
printf.print_format "{}\0A", %144 : i32
// CHECK-NEXT:  printf.print_format "{}\0A", %{{.*}} : i32 {unit}
printf.print_format "{}\0A", %144 : i32 {unit}
// CHECK-NEXT:  "printf.print_char"(%{{.*}}) : (i8) -> ()
"printf.print_char"(%byte) : (i8) -> ()
// CHECK-NEXT:  "printf.print_int"(%{{.*}}) : (i32) -> ()
"printf.print_int"(%12) : (i32) -> ()

// CHECK-GENERIC:      "printf.print_format"() <{format_str = "Hello world!\0A"}> : () -> ()
// CHECK-GENERIC-NEXT: %0 = "test.op"() : () -> i32
// CHECK-GENERIC-NEXT: %1 = "test.op"() : () -> i32
// CHECK-GENERIC-NEXT: %byte = "test.op"() : () -> i8
// CHECK-GENERIC-NEXT: "printf.print_format"(%1, %0, %1, %0) <{format_str = "Uses vals twice {} {} {} {}\0A"}> : (i32, i32, i32, i32) -> ()
// CHECK-GENERIC-NEXT: "printf.print_format"(%0) <{format_str = "{}\0A"}> : (i32) -> ()
// CHECK-GENERIC-NEXT: "printf.print_format"(%0) <{format_str = "{}\0A"}> {unit} : (i32) -> ()
// CHECK-GENERIC-NEXT: "printf.print_char"(%byte) : (i8) -> ()
// CHECK-GENERIC-NEXT: "printf.print_int"(%1) : (i32) -> ()
