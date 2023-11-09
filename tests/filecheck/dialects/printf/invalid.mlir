// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
    printf.print_format "This will fail {}"
}

// CHECK: Operation does not verify: Number of templates in template string must match number of arguments!

// -----
// CHECK: -----

builtin.module {
    %0 = "test.op"() : () -> i32
    printf.print_format "This will fail too {}", %0 : i32, %0 : i32
}

// CHECK: Operation does not verify: Number of templates in template string must match number of arguments!
