// RUN: xdsl-opt %s --print-op-generic | filecheck %s

"builtin.module"() ({

    %arg0, %arg1 = "test.op"() : () -> (i32, i32)
    %icmp_eq_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (i32, i32) -> i1
    %icmp_eq = llvm.icmp "eq" %arg0, %arg1 : i32

    // CHECK: "builtin.module"() ({
    // CHECK-NEXT:  %arg0, %arg1 = "test.op"() : () -> (i32, i32)
    // CHECK-NEXT:  %icmp_eq_p = "llvm.icmp"(%arg0, %arg1) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:  %icmp_eq = "llvm.icmp"(%arg0, %arg1) <{"predicate" = 0 : i64}> : (i32, i32) -> i1
    
}) : () -> ()
