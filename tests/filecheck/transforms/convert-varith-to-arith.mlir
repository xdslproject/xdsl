// RUN: xdsl-opt --split-input-file -p convert-arith-to-varith %s

func.func @test_varith_addi() {
    %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    %1, %2, %3 = "test.op"() : () -> (i32, i32, i32)

    %r = varith.add %a, %b, %c, %1, %2, %3 : i32

    "test.op"(%r) : (i32) -> ()

    return

    // CHECK-LABEL: @test_varith_addi
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %1, %2, %3 = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %x1 = arith.add %a, %b : i32
    // CHECK-NEXT:   %x2 = arith.add %x1, %c : i32
    // CHECK-NEXT:   %x3 = arith.add %x2, %1 : i32
    // CHECK-NEXT:   %x4 = arith.add %x3, %2 : i32
    // CHECK-NEXT:   %x5 = arith.add %x4, %3: i32
    // CHECK-NEXT:   "test.op"(%r) : (i32) -> ()
}
