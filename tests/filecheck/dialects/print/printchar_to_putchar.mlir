// RUN: xdsl-opt -p lower-printchar-to-putchar %s | xdsl-opt | filecheck %s

builtin.module {
    "func.func"() ({
        %n = arith.constant 110 : i32
        %i = arith.constant 105 : i32
        %c = arith.constant 99 : i32
        %e = arith.constant 101 : i32
        %exclamation = arith.constant 33 : i32
        %newline = arith.constant 10 : i32
        %integer = arith.constant 21071830 : i32
	"printf.print_char"(%n) : (i32) -> ()
	"printf.print_char"(%i) : (i32) -> ()
	"printf.print_char"(%c) : (i32) -> ()
	"printf.print_char"(%e) : (i32) -> ()
	"printf.print_char"(%exclamation) : (i32) -> ()
	"printf.print_char"(%newline) : (i32) -> ()
        "printf.print_int"(%integer) : (i32) -> ()
	"printf.print_char"(%newline) : (i32) -> ()
	"func.return"() : () -> ()
    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}

// CHECK: %{{.*}} = arith.constant 52 : i32
// CHECK-NEXT: %{{.*}} = "func.call"(%{{.*}}) {"callee" = @putchar} : (i32) -> i32
// CHECK: func.func private @putchar(i32) -> i32
