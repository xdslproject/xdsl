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

func.func @test_varith_addf() {
    %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)

    %r = varith.add %a, %b, %c, %0, %1, %2: f32

    "test.op"(%r) : (f32) -> ()

    return

    // CHECK-LABEL: @test_varith_addf
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %r = arith.addf %a, %b : f32
    // CHECK-NEXT:   %r_1 = arith.addf %r, %c : f32
    // CHECK-NEXT:   %r_2 = arith.addf %r_1, %0 : f32
    // CHECK-NEXT:   %r_3 = arith.addf %r_2, %1 : f32
    // CHECK-NEXT:   %r_4 = arith.addf %r_3, %2 : f32
    // CHECK-NEXT:   "test.op"(%r_4) : (f32) -> ()
}

func.func @test_varith_mulf() {
    %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)

    %r = varith.mul %a, %b, %c, %0, %1, %2: f32

    "test.op"(%r) : (f32) -> ()

    return

    // CHECK-LABEL: @test_varith_mulf
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %r = arith.mulf %a, %b : f32
    // CHECK-NEXT:   %r_1 = arith.mulf %r, %c : f32
    // CHECK-NEXT:   %r_2 = arith.mulf %r_1, %0 : f32
    // CHECK-NEXT:   %r_3 = arith.mulf %r_2, %1 : f32
    // CHECK-NEXT:   %r_4 = arith.mulf %r_3, %2 : f32
    // CHECK-NEXT:   "test.op"(%r_4) : (f32) -> ()
}
