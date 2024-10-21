// RUN: xdsl-opt --split-input-file -p convert-varith-to-arith %s | filecheck %s

func.func @test_varith_addi() {
    %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    %0, %1, %2 = "test.op"() : () -> (i32, i32, i32)

    %r = varith.add %a, %b, %c, %0, %1, %2: i32

    "test.op"(%r) : (i32) -> ()

    return

    // CHECK-LABEL: @test_varith_addi
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %r = arith.addi %a, %b : i32
    // CHECK-NEXT:   %r_1 = arith.addi %r, %c : i32
    // CHECK-NEXT:   %r_2 = arith.addi %r_1, %0 : i32
    // CHECK-NEXT:   %r_3 = arith.addi %r_2, %1 : i32
    // CHECK-NEXT:   %r_4 = arith.addi %r_3, %2 : i32
    // CHECK-NEXT:   "test.op"(%r_4) : (i32) -> ()
}
