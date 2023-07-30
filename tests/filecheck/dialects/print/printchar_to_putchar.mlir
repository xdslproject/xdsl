// RUN: xdsl-opt -p lower-printchar-to-putchar %s | mlir-opt --convert-math-to-funcs  --test-lower-to-llvm | mlir-cpu-runner --entry-point-result=void | filecheck %s

builtin.module{
    "func.func"() ({
        %n = arith.constant 110 : i32
        %i = arith.constant 105 : i32
        %c = arith.constant 99 : i32
        %e = arith.constant 101 : i32
        %exclamation = arith.constant 33 : i32
        %newline = arith.constant 10 : i32
        %integer = arith.constant -2147483648: i32

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

// CHECK: nice!
// CHECK-NEXT: -2147483648
