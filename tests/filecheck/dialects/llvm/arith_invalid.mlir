// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

"builtin.module"() ({

    %arg0 = "test.op"() : () -> (i32)

    %trunc = llvm.trunc %arg0 : i32 to i64
    // CHECK: invalid cast opcode for cast from i32 to i64

}) : () -> ()

// -----

"builtin.module"() ({

    %arg0 = "test.op"() : () -> (i32)

    %zext = llvm.zext %arg0 : i32 to i16
    // CHECK: invalid cast opcode for cast from i32 to i16
    
}) : () -> ()

// -----

"builtin.module"() ({

    %arg0 = "test.op"() : () -> (i32)

    %sext = llvm.sext %arg0 : i32 to i16
    // CHECK: invalid cast opcode for cast from i32 to i16
    
}) : () -> ()

// -----

"builtin.module"() ({

    %arg0, %arg1 = "test.op"() : () -> (i32, i32)

    %icmp_eq = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (i32, i32) -> vector<7xi1>
    // CHECK: Result must be scalar if operands are scalar, got vector<7xi1>

}) : () -> ()

// -----

"builtin.module"() ({

    %arg0, %arg1 = "test.op"() : () -> (vector<7xi32>, vector<7xi32>)

    %icmp_eq = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (vector<7xi32>, vector<7xi32>) -> i1
    // CHECK: Result must be a vector if operands are vectors, got i1
    
}) : () -> ()
