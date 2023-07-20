// RUN: xdsl-opt -p lower-printchar-to-putchar %s | xdsl-opt | filecheck %s

builtin.module {
    // 4 in ascii
    "func.func"() ({
        %4 = arith.constant 52 : i32
        "printf.print_char"(%4) : (i32) -> ()
        "printf.print_int"(%4) : (i32) -> ()
	"func.return"() : () -> ()
    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}

// CHECK: %{{.*}} = arith.constant 52 : i32
// CHECK-NEXT: %{{.*}} = "func.call"(%{{.*}}) {"callee" = @putchar} : (i32) -> i32
// CHECK: func.func private @putchar(i32) -> i32
