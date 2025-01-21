"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "main", sym_visibility = "private"}> ({
    %0 = "arith.constant"() <{value = 110 : i8}> : () -> i8
    %1 = "arith.constant"() <{value = 105 : i8}> : () -> i8
    %2 = "arith.constant"() <{value = 99 : i8}> : () -> i8
    %3 = "arith.constant"() <{value = 101 : i8}> : () -> i8
    %4 = "arith.constant"() <{value = 33 : i8}> : () -> i8
    %5 = "arith.constant"() <{value = 10 : i8}> : () -> i8
    %6 = "arith.constant"() <{value = -2147483648 : i32}> : () -> i32
    "printf.print_char"(%0) : (i8) -> ()
    "printf.print_char"(%1) : (i8) -> ()
    "printf.print_char"(%2) : (i8) -> ()
    "printf.print_char"(%3) : (i8) -> ()
    "printf.print_char"(%4) : (i8) -> ()
    "printf.print_char"(%5) : (i8) -> ()
    "printf.print_int"(%6) : (i32) -> ()
    "printf.print_char"(%5) : (i8) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
