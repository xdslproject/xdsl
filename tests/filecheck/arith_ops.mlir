// RUN: ../../src/xdsl/xdsl_opt.py %s | ../../src/xdsl/xdsl_opt.py | filecheck %s

module() {
  func.func() ["sym_name" = "main", "function_type" = !fun<[], []>, "sym_visibility" = "private"]{
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.constant() ["value" = 42 : !i32]
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32) 
    %3 : !i32 = arith.subi(%0 : !i32, %1 : !i32) 
    %4 : !i32 = arith.muli(%0 : !i32, %1 : !i32) 
    %5 : !i32 = arith.floordivsi(%0 : !i32, %1 : !i32) 
    %6 : !i32 = arith.remsi(%0 : !i32, %1 : !i32) 
    func.return()
  }
}

// CHECK: func.func() ["sym_name" = "main"
// CHECK-NEXT: %{{.*}} : !i32 = arith.constant() ["value" = 42 : !i32] 
// CHECK-NEXT: %{{.*}} : !i32 = arith.constant() ["value" = 42 : !i32]
// CHECK-NEXT: %{{.*}} : !i32 = arith.addi(%{{.*}} : !i32, %{{.*}} : !i32) 
// CHECK-NEXT: %{{.*}} : !i32 = arith.subi(%{{.*}} : !i32, %{{.*}} : !i32) 
// CHECK-NEXT: %{{.*}} : !i32 = arith.muli(%{{.*}} : !i32, %{{.*}} : !i32) 
// CHECK-NEXT: %{{.*}} : !i32 = arith.floordivsi(%{{.*}} : !i32, %{{.*}} : !i32)
// CHECK-NEXT: %{{.*}} : !i32 = arith.remsi(%{{.*}} : !i32, %{{.*}} : !i32) 

