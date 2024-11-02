// RUN: xdsl-opt --split-input-file -p varith-fuse-repeated-operands %s | filecheck %s

func.func @test_addi() {
    %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    %1, %2, %3 = "test.op"() : () -> (i32, i32, i32)

    %r = varith.add %a, %b, %a, %a, %b, %c : i32

    "test.op"(%r) : (i32) -> ()

    return

    // CHECK-LABEL: @test_addi
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %3 = arith.constant 3 : i32
    // CHECK-NEXT:   %4 = arith.constant 2 : i32
    // CHECK-NEXT:   %5 = arith.muli %3, %a : i32
    // CHECK-NEXT:   %6 = arith.muli %4, %b : i32
    // CHECK-NEXT:   %r = varith.add %5, %6, %c : i32
    // CHECK-NEXT:   "test.op"(%r) : (i32) -> ()
}

func.func @test_addf() {
    %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    %1, %2, %3 = "test.op"() : () -> (f32, f32, f32)

    %r = varith.add %a, %b, %a, %a, %b, %c : f32

    "test.op"(%r) : (f32) -> ()

    return

    // CHECK-LABEL: @test_addf
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %3 = arith.constant 3.000000e+00 : f32
    // CHECK-NEXT:   %4 = arith.constant 2.000000e+00 : f32
    // CHECK-NEXT:   %5 = arith.mulf %3, %a : f32
    // CHECK-NEXT:   %6 = arith.mulf %4, %b : f32
    // CHECK-NEXT:   %r = varith.add %5, %6, %c : f32
    // CHECK-NEXT:   "test.op"(%r) : (f32) -> ()
}
