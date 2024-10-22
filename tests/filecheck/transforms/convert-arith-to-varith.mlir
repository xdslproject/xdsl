// RUN: xdsl-opt --split-input-file -p convert-arith-to-varith %s | filecheck %s

func.func @test_addi() {
    %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    %1, %2, %3 = "test.op"() : () -> (i32, i32, i32)

    %x1 = arith.addi %a, %b : i32
    %y1 = arith.addi %x1, %c : i32

    %x2 = arith.addi %1, %2 : i32
    %y2 = arith.addi %x2, %3 : i32

    %r = varith.add %y1, %y2 : i32

    "test.op"(%r, %x2) : (i32, i32) -> ()

    return

    // CHECK-LABEL: @test_addi
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (i32, i32, i32)
    // CHECK-NEXT:   %x2 = arith.addi %0, %1 : i32
    // CHECK-NEXT:   %r = varith.add %a, %b, %c, %0, %1, %2 : i32
    // CHECK-NEXT:   "test.op"(%r, %x2) : (i32, i32) -> ()
}


func.func @test_addf() {
    %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    %1, %2, %3 = "test.op"() : () -> (f32, f32, f32)

    %x1 = arith.addf %a, %b : f32
    %y1 = arith.addf %x1, %c : f32

    %x2 = arith.addf %1, %2 : f32
    %y2 = arith.addf %x2, %3 : f32

    %r = varith.add %y1, %y2 : f32

    "test.op"(%r, %x2) : (f32, f32) -> ()

    return

    // CHECK-LABEL: @test_addf
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %x2 = arith.addf %0, %1 : f32
    // CHECK-NEXT:   %r = varith.add %a, %b, %c, %0, %1, %2 : f32
    // CHECK-NEXT:   "test.op"(%r, %x2) : (f32, f32) -> ()
}

func.func @test_mulf() {
    %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    %1, %2, %3 = "test.op"() : () -> (f32, f32, f32)

    %x1 = arith.mulf %a, %b : f32
    %y1 = arith.mulf %x1, %c : f32

    %x2 = arith.mulf %1, %2 : f32
    %y2 = arith.mulf %x2, %3 : f32

    %r = varith.mul %y1, %y2 : f32

    "test.op"(%r, %x2) : (f32, f32) -> ()

    return

    // CHECK-LABEL: @test_mulf
    // CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)
    // CHECK-NEXT:   %x2 = arith.mulf %0, %1 : f32
    // CHECK-NEXT:   %r = varith.mul %a, %b, %c, %0, %1, %2 : f32
    // CHECK-NEXT:   "test.op"(%r, %x2) : (f32, f32) -> ()
}

func.func @test() {
    %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (f32, f32, f32, f32, f32, f32)
    %6 = arith.constant 1.234500e-01 : f32
    %a = arith.addf %5, %4 : f32
    %b = arith.addf %a, %3 : f32
    %c = arith.addf %b, %2 : f32
    %d = arith.addf %c, %1 : f32
    %e = arith.addf %d, %0 : f32
    %12 = arith.mulf %e, %6 : f32
    "test.op"(%12) : (f32) -> ()
    func.return

    // CHECK-LABEL: @test
    // CHECK-NEXT:   %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (f32, f32, f32, f32, f32, f32)
    // CHECK-NEXT:   %6 = arith.constant 1.234500e-01 : f32
    // CHECK-NEXT:   %e = varith.add %5, %4, %3, %2, %1, %0 : f32
    // CHECK-NEXT:   %7 = arith.mulf %e, %6 : f32
    // CHECK-NEXT:   "test.op"(%7) : (f32) -> ()
}

func.func @test2() {
    %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (f32, f32, f32, f32, f32, f32)
    %6 = arith.constant 1.234500e-01 : f32
    %a = arith.addf %5, %4 : f32
    %b = arith.addf %3, %a : f32
    %c = arith.addf %2, %b : f32
    %d = arith.addf %1, %c : f32
    %e = arith.addf %0, %d : f32
    %12 = arith.mulf %e, %6 : f32
    "test.op"(%12) : (f32) -> ()
    func.return

    // CHECK-LABEL: @test
    // CHECK-NEXT:   %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (f32, f32, f32, f32, f32, f32)
    // CHECK-NEXT:   %6 = arith.constant 1.234500e-01 : f32
    // CHECK-NEXT:   %e = varith.add %1, %2, %3, %5, %4, %0 : f32
    // CHECK-NEXT:   %7 = arith.mulf %e, %6 : f32
    // CHECK-NEXT:   "test.op"(%7) : (f32) -> ()
}
