// RUN: xdsl-opt -p printf-to-putchar %s | mlir-opt --convert-math-to-funcs --convert-scf-to-cf --convert-to-llvm | mlir-runner --entry-point-result=void | filecheck %s

builtin.module{
    "func.func"() ({
        %n = arith.constant 110 : i8
        %i = arith.constant 105 : i8
        %c = arith.constant 99 : i8
        %e = arith.constant 101 : i8
        %exclamation = arith.constant 33 : i8
        %newline = arith.constant 10 : i8
        %integer = arith.constant -2147483648: i32

	"printf.print_char"(%n) : (i8) -> ()
	"printf.print_char"(%i) : (i8) -> ()
	"printf.print_char"(%c) : (i8) -> ()
	"printf.print_char"(%e) : (i8) -> ()
	"printf.print_char"(%exclamation) : (i8) -> ()
	"printf.print_char"(%newline) : (i8) -> ()
    "printf.print_int"(%integer) : (i32) -> ()
	"printf.print_char"(%newline) : (i8) -> ()
	"func.return"() : () -> ()
    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}

// CHECK: nice!
// CHECK-NEXT: -2147483648
