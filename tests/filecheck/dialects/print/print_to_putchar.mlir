// RUN: xdsl-opt %s -p print-to-putchar | filecheck %s

builtin.module {
    "func.func"() ({
        %12 = "arith.constant"() {value = 12 : i32} : () -> i32

        print.println "Hello wordl {}!", %12 : i32

        "func.return"() : () -> ()
    }) {sym_name = "main", function_type=() -> ()} : () -> ()
}
