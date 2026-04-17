// RUN: xdsl-opt -p convert-arith-to-riscv --split-input-file --verify-diagnostics %s | filecheck %s

// CHECK: Integer vector constants cannot be lowered to float registers
builtin.module {
    %dense_8xi8 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : vector<8xi8>
}

// -----

// CHECK: Integer vector constants cannot be lowered to float registers
builtin.module {
    %dense_4xi16 = arith.constant dense<[1, 2, 3, 4]> : vector<4xi16>
}

// -----

// CHECK: Integer vector constants cannot be lowered to float registers
builtin.module {
    %dense_2xi32 = arith.constant dense<[1, 2]> : vector<2xi32>
}

// -----

// CHECK: Integer vector constants cannot be lowered to float registers
builtin.module {
    %dense_1xi64 = arith.constant dense<[42]> : vector<1xi64>
}
