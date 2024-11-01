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
