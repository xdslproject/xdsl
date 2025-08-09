// RUN: xdsl-opt %s -p reconcile-unrealized-casts | filecheck %s

builtin.module {
    // CHECK:  builtin.module {

    func.func @unused_cast(%arg0: i64) -> i64 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i16
        "func.return"(%arg0) : (i64) -> ()
    }

    // CHECK-NEXT:  func.func @unused_cast(%{{.*}} : i64) -> i64 {
    // CHECK-NEXT:    func.return %{{.*}} : i64
    // CHECK-NEXT:  }

    // -----

    func.func @simple_cycle(%arg0: i64) -> i64 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
        "func.return"(%1) : (i64) -> ()
    }

    // CHECK-NEXT:  func.func @simple_cycle(%{{.*}} : i64) -> i64 {
    // CHECK-NEXT:    func.return %{{.*}} : i64
    // CHECK-NEXT:  }

    // -----

    func.func @cycle_singleblock(%arg0: i64) -> i64 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i16
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i16) -> i1
        %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
        %3 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i16
        %4 = "builtin.unrealized_conversion_cast"(%3) : (i16) -> i64
        %5 = "test.op"(%2, %4) : (i64, i64) -> i64
        func.return %5 : i64
    }

    // CHECK-NEXT:    func.func @cycle_singleblock(%{{.*}} : i64) -> i64 {
    // CHECK-NEXT:      %0 = "test.op"(%{{.*}}, %{{.*}}) : (i64, i64) -> i64
    // CHECK-NEXT:      func.return %0 : i64
    // CHECK-NEXT:    }

    // -----

    func.func @cycle_multiblock(%arg0: i64) -> i64 {
        %c0 = "test.op"() : () -> i32
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i16
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i16) -> i1
        %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
        %3 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i16
        cf.br ^bb1(%c0 : i32)
        ^bb1(%33: i32):  // pred: ^bb0
        %4 = "builtin.unrealized_conversion_cast"(%3) : (i16) -> i64
        %5 = "test.op"(%2, %4) : (i64, i64) -> i64
        func.return %5 : i64
    }

    // CHECK-NEXT:    func.func @cycle_multiblock(%{{.*}} : i64) -> i64 {
    // CHECK-NEXT:      %c0 = "test.op"() : () -> i32
    // CHECK-NEXT:      cf.br ^bb0(%c0 : i32)
    // CHECK-NEXT:    ^bb0(%0 : i32):
    // CHECK-NEXT:      %1 = "test.op"(%{{.*}}, %{{.*}}) : (i64, i64) -> i64
    // CHECK-NEXT:      func.return %1 : i64
    // CHECK-NEXT:    }

    // -----

    func.func @failure_simple_cast(%arg0: i64) -> i32 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
        func.return %0 : i32
    }

    // CHECK-NEXT:    func.func @failure_simple_cast(%{{.*}} : i64) -> i32 {
    // CHECK-NEXT:      %0 = builtin.unrealized_conversion_cast %{{.*}} : i64 to i32
    // CHECK-NEXT:      func.return %0 : i32
    // CHECK-NEXT:    }

    // -----

    func.func @failure_chain(%arg0: i64) -> i32 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i1
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i1) -> i32
        func.return %1 : i32
    }

    // CHECK-NEXT:    func.func @failure_chain(%{{.*}} : i64) -> i32 {
    // CHECK-NEXT:      %0 = builtin.unrealized_conversion_cast %{{.*}} : i64 to i1
    // CHECK-NEXT:      %1 = builtin.unrealized_conversion_cast %0 : i1 to i32
    // CHECK-NEXT:      func.return %1 : i32
    // CHECK-NEXT:    }


    func.func @cycle_singleblock_var_ops(%arg0: i64, %arg1: i64) -> i64 {
        %0, %1 = "builtin.unrealized_conversion_cast"(%arg0, %arg1) : (i64, i64) -> (i16, i16)
        %2, %3 = "builtin.unrealized_conversion_cast"(%0, %1) : (i16, i16) -> (i1, i1)
        %4, %5 = "builtin.unrealized_conversion_cast"(%2, %3) : (i1, i1) -> (i64, i64)
        %6, %7 = "builtin.unrealized_conversion_cast"(%2, %3) : (i1, i1) -> (i16, i16)
        %8, %9 = "builtin.unrealized_conversion_cast"(%6, %7) : (i16, i16) -> (i64, i64)
        %10 = "test.op"(%4, %5, %8, %9) : (i64, i64, i64, i64) -> i64
        func.return %10 : i64
    }

    // CHECK-NEXT:    func.func @cycle_singleblock_var_ops(%{{.*}} : i64, %{{.*}} : i64) -> i64 {
    // CHECK-NEXT:      %0 = "test.op"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64) -> i64
    // CHECK-NEXT:      func.return %0 : i64
    // CHECK-NEXT:    }

    func.func @mismatch_size_cast_use(%arg0: i64, %arg1: i64) -> i64 {
        %0, %1 = "builtin.unrealized_conversion_cast"(%arg0, %arg1) : (i64, i64) -> (i16, i16)
        %3 = "builtin.unrealized_conversion_cast"(%0) : (i16) -> (i1)
        %10 = "test.op"(%3, %3) : (i1, i1) -> i64
        func.return %10 : i64
    }

    // CHECK-NEXT:    func.func @mismatch_size_cast_use(%{{.*}} : i64, %{{.*}} : i64) -> i64 {
    // CHECK-NEXT:      %0, %1 = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} : i64, i64 to i16, i16
    // CHECK-NEXT:      %2 = builtin.unrealized_conversion_cast %0 : i16 to i1
    // CHECK-NEXT:      %3 = "test.op"(%2, %2) : (i1, i1) -> i64
    // CHECK-NEXT:      func.return %3 : i64
    // CHECK-NEXT:    }

}
// CHECK-NEXT:  }
